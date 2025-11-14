#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import json
import torch
import argparse
import numpy as np

from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E


class AVSR(torch.nn.Module):
    def __init__(self, modality, model_path, model_conf, rnnlm=None, rnnlm_conf=None,
        penalty=0., ctc_weight=0.1, lm_weight=0., beam_size=40, device="cuda:0"):
        super(AVSR, self).__init__()
        self.device = device

        if modality == "audiovisual":
            from espnet.nets.pytorch_backend.e2e_asr_transformer_av import E2E
        else:
            from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        self.train_args = argparse.Namespace(**args)

        labels_type = getattr(self.train_args, "labels_type", "char")
        if labels_type == "char":
            self.token_list = self.train_args.char_list
        elif labels_type == "unigram5000":
            file_path = os.path.join(os.path.dirname(__file__), "tokens", "unigram5000_units.txt")
            # use utf-8 encoding to avoid Windows cp1252 encoding issues
            with open(file_path, 'r', encoding='utf-8') as f:
                self.token_list = ['<blank>'] + [word.split()[0] for word in f.read().splitlines()] + ['<eos>']
        self.odim = len(self.token_list)

        self.model = E2E(self.odim, self.train_args)
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.to(device=self.device).eval()

        # convert penalty to int for type compatibility (though float works at runtime)
        penalty_int = int(penalty) if penalty is not None else 0
        self.beam_search = get_beam_search_decoder(self.model, self.token_list, rnnlm, rnnlm_conf, penalty_int, ctc_weight, lm_weight, beam_size)
        self.beam_search.to(device=self.device).eval()
        
    def infer(self, data):
        with torch.no_grad():
            if isinstance(data, tuple):
                # audiovisual mode - encode takes (x, aux_x)
                enc_feats = self.model.encode(data[0].to(self.device), data[1].to(self.device))  # type: ignore
            else:
                # video-only mode - encode takes only x (no aux_x parameter)
                enc_feats = self.model.encode(data.to(self.device))  # type: ignore
            nbest_hyps = self.beam_search(enc_feats)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            transcription = add_results_to_json(nbest_hyps, self.token_list)
            transcription = transcription.replace("▁", " ").strip()
        return transcription.replace("<eos>", "")
    
    def infer_with_alignment(self, data, video_fps=25.0):
        """
        Infer transcription with CTC forced alignment for subtitle generation
        
        Args:
            data: Input video data (tensor or tuple)
            video_fps: Video frame rate (default: 25.0 fps for AutoAVSR standard)
            
        Returns:
            dict with keys:
                - 'transcription': str - The transcription text
                - 'word_timestamps': list - List of dicts with 'word', 'start', 'end' keys
                - 'frame_alignments': list - Frame-level token alignments (for debugging)
        """
        with torch.no_grad():
            # encode video to get features
            # NOTE: encode() returns 2D tensor (T, D) because it squeezes batch dimension
            if isinstance(data, tuple):
                # audiovisual mode - encode takes (x, aux_x)
                enc_feats = self.model.encode(data[0].to(self.device), data[1].to(self.device))  # type: ignore
            else:
                # video-only mode - encode takes only x (no aux_x parameter)
                enc_feats = self.model.encode(data.to(self.device))  # type: ignore
            
            # ensure enc_feats is in correct format for alignment
            # encode() may return (T, D) or (B, T, D) depending on model
            
            # get transcription using beam search
            nbest_hyps = self.beam_search(enc_feats)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            transcription = add_results_to_json(nbest_hyps, self.token_list)
            transcription = transcription.replace("▁", " ").strip()
            transcription = transcription.replace("<eos>", "")
            
            # extract token IDs from best hypothesis for alignment
            if nbest_hyps and 'yseq' in nbest_hyps[0]:
                token_ids = nbest_hyps[0]['yseq']
                # remove SOS and EOS tokens
                token_ids = [t for t in token_ids if t != self.odim - 1]  # remove EOS
                if token_ids and token_ids[0] == self.odim - 1:
                    token_ids = token_ids[1:]  # remove SOS if present
            else:
                # fallback: convert transcription to token IDs
                token_ids = []
                words = transcription.split()
                for word in words:
                    # try to find word in token list (simplified - may need improvement)
                    word_lower = word.lower()
                    if word_lower in self.token_list:
                        token_ids.append(self.token_list.index(word_lower))
                    else:
                        # split into characters if word not found
                        for char in word:
                            if char in self.token_list:
                                token_ids.append(self.token_list.index(char))
            
            # perform CTC forced alignment if CTC module is available
            word_timestamps = []
            frame_alignments = []
            
            if hasattr(self.model, 'ctc') and self.model.ctc is not None and len(token_ids) > 0:
                try:
                    # use CTC forced_align method for accurate frame-level alignment
                    # enc_feats might be (T, D) or (B, T, D) - handle both cases
                    # forced_align can handle both, but we need to determine T correctly
                    if len(enc_feats.shape) == 3:
                        # has batch dimension (B, T, D)
                        if enc_feats.shape[0] == 1:
                            # single batch - extract T from second dimension
                            T = enc_feats.shape[1]
                            enc_feats_for_align = enc_feats  # keep as (1, T, D)
                        else:
                            # multiple batches - use first batch
                            T = enc_feats.shape[1]
                            enc_feats_for_align = enc_feats[0:1]  # take first batch (1, T, D)
                    elif len(enc_feats.shape) == 2:
                        # 2D (T, D) - forced_align will add batch dimension
                        T = enc_feats.shape[0]
                        enc_feats_for_align = enc_feats  # pass as (T, D) - forced_align handles it
                    else:
                        raise ValueError(f"Unexpected enc_feats shape: {enc_feats.shape}, expected 2D (T, D) or 3D (B, T, D)")
                    
                    # ensure enc_feats is on correct device
                    if enc_feats_for_align.device != self.device:
                        enc_feats_for_align = enc_feats_for_align.to(self.device)
                    
                    # convert token_ids to numpy array for forced_align
                    if len(token_ids) == 0:
                        raise ValueError("Empty token_ids - cannot perform alignment")
                    
                    token_array = np.array(token_ids, dtype=np.int64)
                    
                    # perform CTC forced alignment
                    # forced_align handles both 2D (T, D) and 3D (B, T, D) inputs
                    aligned_frames = self.model.ctc.forced_align(enc_feats_for_align, token_array, blank_id=0)
                    
                    # aligned_frames is a list of token IDs, one per frame
                    # map tokens to words from transcription using CTC alignment
                    if len(aligned_frames) > 0 and len(token_ids) > 0:
                        # get transcription words
                        words = transcription.split()
                        
                        # create mapping: token_id -> word index (assuming sequential mapping)
                        # tokens in token_ids correspond to words in transcription
                        token_to_word_map = {}
                        if len(token_ids) == len(words):
                            # direct mapping: one token per word
                            for i, token_id in enumerate(token_ids):
                                if i < len(words):
                                    token_to_word_map[token_id] = i
                        else:
                            # approximate mapping: distribute tokens across words
                            tokens_per_word = len(token_ids) / len(words) if words else 0
                            for i, token_id in enumerate(token_ids):
                                word_idx = int(i / tokens_per_word) if tokens_per_word > 0 else 0
                                if word_idx < len(words):
                                    token_to_word_map[token_id] = word_idx
                        
                        # group consecutive frames with same token to create word timestamps
                        current_token = aligned_frames[0]
                        start_frame = 0
                        
                        for frame_idx, token_id in enumerate(aligned_frames):
                            if token_id != current_token or frame_idx == len(aligned_frames) - 1:
                                # token changed or reached end - create timestamp for previous token
                                if current_token != 0 and current_token < len(self.token_list):  # skip blank tokens
                                    end_frame = frame_idx if token_id != current_token else frame_idx + 1
                                    
                                    # convert frames to timestamps
                                    start_time = start_frame / video_fps
                                    end_time = min(end_frame / video_fps, T / video_fps)
                                    
                                    # map token to word
                                    if current_token in token_to_word_map:
                                        word_idx = token_to_word_map[current_token]
                                        if word_idx < len(words):
                                            word = words[word_idx]
                                            # check if we already have a timestamp for this word (merge if overlapping)
                                            existing_idx = None
                                            for i, ts in enumerate(word_timestamps):
                                                if ts['word'] == word:
                                                    # merge timestamps (extend end time)
                                                    word_timestamps[i]['end'] = max(word_timestamps[i]['end'], end_time)
                                                    existing_idx = i
                                                    break
                                            
                                            if existing_idx is None:
                                                word_timestamps.append({
                                                    'word': word,
                                                    'start': start_time,
                                                    'end': end_time
                                                })
                                    
                                    frame_alignments.append({
                                        'frame_start': start_frame,
                                        'frame_end': end_frame,
                                        'token_id': current_token,
                                        'token': self.token_list[current_token] if current_token < len(self.token_list) else ''
                                    })
                                
                                # start new token segment
                                current_token = token_id
                                start_frame = frame_idx
                        
                        # sort word timestamps by start time and merge overlapping/adjacent words
                        if word_timestamps:
                            word_timestamps.sort(key=lambda x: x['start'])
                            # merge adjacent words with same or overlapping timestamps
                            merged_timestamps = []
                            for ts in word_timestamps:
                                if merged_timestamps and ts['start'] <= merged_timestamps[-1]['end']:
                                    # merge with previous
                                    merged_timestamps[-1]['end'] = max(merged_timestamps[-1]['end'], ts['end'])
                                else:
                                    merged_timestamps.append(ts)
                            word_timestamps = merged_timestamps
                    else:
                        # fallback: use simple word-level distribution
                        words = transcription.split()
                        if words:
                            time_per_word = (T / video_fps) / len(words)
                            for i, word in enumerate(words):
                                word_timestamps.append({
                                    'word': word,
                                    'start': i * time_per_word,
                                    'end': (i + 1) * time_per_word
                                })
                
                except Exception as e:
                    print(f"[ALIGNMENT] WARNING: CTC forced alignment failed: {e}, using fallback")
                    import traceback
                    traceback.print_exc()
                    # fallback: simple word-level distribution
                    words = transcription.split()
                    if words:
                        T = enc_feats.shape[0] if hasattr(enc_feats, 'shape') else 75
                        time_per_word = (T / video_fps) / len(words)
                        for i, word in enumerate(words):
                            word_timestamps.append({
                                'word': word,
                                'start': i * time_per_word,
                                'end': (i + 1) * time_per_word
                            })
            else:
                # no CTC module or no tokens - use simple word-level distribution
                words = transcription.split()
                if words:
                    T = enc_feats.shape[0] if hasattr(enc_feats, 'shape') else 75  # default 3 seconds at 25fps
                    time_per_word = (T / video_fps) / len(words)
                    for i, word in enumerate(words):
                        word_timestamps.append({
                            'word': word,
                            'start': i * time_per_word,
                            'end': (i + 1) * time_per_word
                        })
            
            return {
                'transcription': transcription,
                'word_timestamps': word_timestamps,
                'frame_alignments': frame_alignments
            }


def get_beam_search_decoder(model, token_list, rnnlm=None, rnnlm_conf=None, penalty=0, ctc_weight=0.1, lm_weight=0., beam_size=40):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        # get_model_conf returns argparse.Namespace for LM (not tuple)
        if isinstance(lm_args, tuple):
            # if it's a tuple (for ASR), extract the args part
            _, _, lm_args = lm_args
        lm_model_module = getattr(lm_args, "model_module", "default")
        backend = getattr(lm_args, "backend", "pytorch")
        lm_class = dynamic_import_lm(lm_model_module, backend)
        lm = lm_class(len(token_list), lm_args)  # type: ignore
        torch_load(rnnlm, lm)
        if hasattr(lm, 'eval'):
            lm.eval()  # type: ignore

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        length_bonus=penalty,
    )

    # determine pre_beam_score_key - use "decoder" unless ctc_weight is 1.0
    pre_beam_key: str | None = None if ctc_weight == 1.0 else "decoder"
    
    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=pre_beam_key,  # type: ignore
    )
