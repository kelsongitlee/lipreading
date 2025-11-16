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
            
            # get transcription using beam search
            nbest_hyps = self.beam_search(enc_feats)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            transcription = add_results_to_json(nbest_hyps, self.token_list)
            transcription = transcription.replace("▁", " ").strip()
            transcription = transcription.replace("<eos>", "")
            
            # extract token IDs from best hypothesis for alignment
            # CRITICAL FIX: Properly extract and clean token IDs from beam search result
            if nbest_hyps and 'yseq' in nbest_hyps[0]:
                yseq = nbest_hyps[0]['yseq']
                # Convert to list if tensor
                if isinstance(yseq, torch.Tensor):
                    token_ids = yseq.cpu().tolist()
                else:
                    token_ids = list(yseq)
                
                # Remove SOS (first token, usually odim-1) and EOS (last token, odim-1)
                sos_eos_id = self.odim - 1
                # Remove EOS tokens
                token_ids = [t for t in token_ids if t != sos_eos_id]
                # Remove SOS if present at start
                if token_ids and token_ids[0] == sos_eos_id:
                    token_ids = token_ids[1:]
                # Remove any remaining SOS/EOS in middle (shouldn't happen, but safety check)
                token_ids = [t for t in token_ids if t < len(self.token_list) and t >= 0]
            else:
                # fallback: convert transcription to token IDs (should not happen with proper beam search)
                token_ids = []
                # Use the transcription tokens directly by finding them in token_list
                # This is less accurate but provides fallback
                transcription_tokens = transcription.replace("▁", " ▁").split()
                for token_text in transcription_tokens:
                    token_clean = token_text.replace("▁", "").strip()
                    if not token_clean or token_clean in ['<blank>', '<eos>', '<sos>']:
                        continue
                    # Try exact match first (with ▁ prefix for sentencepiece)
                    token_with_prefix = "▁" + token_clean
                    if token_with_prefix in self.token_list:
                        token_ids.append(self.token_list.index(token_with_prefix))
                    elif token_clean in self.token_list:
                        token_ids.append(self.token_list.index(token_clean))
                    else:
                        # Last resort: try lowercase
                        if token_clean.lower() in self.token_list:
                            token_ids.append(self.token_list.index(token_clean.lower()))
            
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
                        enc_feats_for_align = enc_feats  # pass as (T, D)
                    else:
                        raise ValueError(f"Unexpected enc_feats shape: {enc_feats.shape}, expected 2D (T, D) or 3D (B, T, D)")
                    
                    # convert token_ids to numpy array for forced_align
                    token_array = np.array(token_ids, dtype=np.int64)
                    
                    # perform CTC forced alignment
                    # forced_align will handle the batch dimension internally
                    aligned_frames = self.model.ctc.forced_align(enc_feats_for_align, token_array, blank_id=0)
                    
                    # aligned_frames is a list of token IDs, one per frame
                    # CRITICAL FIX: Reconstruct full words from sentencepiece tokens
                    # Sentencepiece tokens: ▁ prefix = word start, no prefix = continuation
                    # Group tokens to form complete words
                    if len(aligned_frames) > 0:
                        # First pass: collect all token segments with their frame ranges
                        # CRITICAL FIX: Properly handle all tokens including the last one
                        token_segments = []
                        current_token = aligned_frames[0]
                        start_frame = 0
                        
                        for frame_idx, token_id in enumerate(aligned_frames):
                            # Check if token changed
                            token_changed = (token_id != current_token)
                            is_last_frame = (frame_idx == len(aligned_frames) - 1)
                            
                            if token_changed:
                                # Token changed - finish current segment
                                if current_token != 0 and current_token < len(self.token_list):  # skip blank tokens
                                    end_frame = frame_idx  # current segment ends at previous frame
                                    token_text = self.token_list[current_token]
                                    token_segments.append({
                                        'token_id': current_token,
                                        'token_text': token_text,
                                        'start_frame': start_frame,
                                        'end_frame': end_frame
                                    })
                                
                                # Start new token segment
                                current_token = token_id
                                start_frame = frame_idx
                            
                            # Handle last frame separately to ensure we capture the final token
                            if is_last_frame:
                                # Add the final token segment (only if not blank)
                                if current_token != 0 and current_token < len(self.token_list):  # skip blank tokens
                                    end_frame = frame_idx + 1  # end_frame is exclusive, so +1
                                    token_text = self.token_list[current_token]
                                    
                                    # Check if this token segment wasn't already added (happens if token changed on last frame)
                                    already_added = False
                                    if token_segments:
                                        last_seg = token_segments[-1]
                                        if (last_seg['token_id'] == current_token and 
                                            last_seg['start_frame'] == start_frame and
                                            last_seg['end_frame'] >= frame_idx):
                                            already_added = True
                                    
                                    if not already_added:
                                        token_segments.append({
                                            'token_id': current_token,
                                            'token_text': token_text,
                                            'start_frame': start_frame,
                                            'end_frame': end_frame
                                        })
                        
                        # Second pass: group tokens into words based on ▁ prefix
                        # CRITICAL FIX: Properly handle word boundaries for sentencepiece tokens
                        # Tokens with ▁ prefix = word start, tokens without ▁ = continuation of previous word
                        current_word_tokens = []
                        current_word_start_frame = None
                        
                        for seg in token_segments:
                            token_text = seg['token_text']
                            has_word_boundary = token_text.startswith('▁') or '▁' in token_text
                            
                            # Clean token text (remove special markers and ▁ prefix)
                            clean_token = token_text.replace('<blank>', '').replace('<eos>', '').replace('<sos>', '').replace('▁', '').strip()
                            
                            # Skip empty or special tokens
                            if not clean_token or clean_token in ['<blank>', '<eos>', '<sos>']:
                                continue
                            
                            # CRITICAL FIX: If we see a word boundary marker, finish current word (if any) and start new one
                            if has_word_boundary:
                                # Finish current word if we have one
                                if current_word_tokens and current_word_start_frame is not None:
                                    full_word = ''.join([t['text'] for t in current_word_tokens])
                                    if full_word:
                                        start_time = current_word_start_frame / video_fps
                                        end_time = current_word_tokens[-1]['end_frame'] / video_fps
                                        word_timestamps.append({
                                            'word': full_word,
                                            'start': start_time,
                                            'end': end_time
                                        })
                                
                                # Start new word (with word boundary marker)
                                current_word_tokens = [{'text': clean_token, 'end_frame': seg['end_frame']}]
                                current_word_start_frame = seg['start_frame']
                            else:
                                # No word boundary = continuation of current word
                                if not current_word_tokens:
                                    # This shouldn't happen (first token should have ▁), but handle it
                                    current_word_start_frame = seg['start_frame']
                                current_word_tokens.append({'text': clean_token, 'end_frame': seg['end_frame']})
                        
                        # Finish last word (important: don't forget the final word!)
                        if current_word_tokens and current_word_start_frame is not None:
                            full_word = ''.join([t['text'] for t in current_word_tokens])
                            if full_word:
                                start_time = current_word_start_frame / video_fps
                                end_time = current_word_tokens[-1]['end_frame'] / video_fps
                                word_timestamps.append({
                                    'word': full_word,
                                    'start': start_time,
                                    'end': end_time
                                })
                        
                        # Also populate frame_alignments for debugging
                        for seg in token_segments:
                            token_text = seg['token_text']
                            clean_token = token_text.replace('<blank>', '').replace('<eos>', '').replace('<sos>', '').replace('▁', '').strip()
                            frame_alignments.append({
                                'frame_start': seg['start_frame'],
                                'frame_end': seg['end_frame'],
                                'token_id': seg['token_id'],
                                'token': clean_token
                            })
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
