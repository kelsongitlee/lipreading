#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import json
import torch
import argparse
import numpy as np
import re

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
            # CRITICAL: Clear GPU cache before encoding to free up memory
            # The encode() step is memory-intensive and may trigger OOM if cache is fragmented
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # encode video to get features
            # NOTE: encode() returns 2D tensor (T, D) because it squeezes batch dimension
            if isinstance(data, tuple):
                # audiovisual mode - encode takes (x, aux_x)
                enc_feats = self.model.encode(data[0].to(self.device), data[1].to(self.device))  # type: ignore
            else:
                # video-only mode - encode takes only x (no aux_x parameter)
                enc_feats = self.model.encode(data.to(self.device))  # type: ignore
            
            # get transcription using beam search
            # DEBUG: Log encoded features before beam search
            enc_frame_count = enc_feats.shape[0] if hasattr(enc_feats, 'shape') else 0
            print(f"[MODEL] Before beam search: {enc_frame_count} encoded frames")
            
            # Original approach: Use default beam search call (maxlenratio=0.0)
            # This uses early stopping via end_detect() which is more memory-efficient
            # For video upload with high accuracy, the model should still complete transcription
            nbest_hyps = self.beam_search(enc_feats)
            # forward() returns List[Hypothesis], convert to list of dicts for consistency
            nbest_hyps = [h.asdict() if hasattr(h, 'asdict') else h for h in nbest_hyps] if nbest_hyps else []
            
            # DEBUG: Log beam search results
            if nbest_hyps:
                best_hyp = nbest_hyps[0] if isinstance(nbest_hyps, list) else nbest_hyps
                # best_hyp is already a dict (from asdict conversion above)
                yseq = best_hyp.get('yseq', []) if isinstance(best_hyp, dict) else []
                yseq_length = len(yseq) if yseq else 0
                print(f"[MODEL] Beam search result: {len(nbest_hyps) if isinstance(nbest_hyps, list) else 1} hypotheses, best has {yseq_length} tokens (max possible: {enc_frame_count})")
                if yseq_length < enc_frame_count:
                    print(f"[MODEL] WARNING: Beam search stopped early! Generated {yseq_length} tokens but had {enc_frame_count} frames available!")
            else:
                print(f"[MODEL] ERROR: Beam search returned no hypotheses!")
            
            # nbest_hyps is already a list of dicts from line 108, just take first one
            nbest_hyps = nbest_hyps[:1] if isinstance(nbest_hyps, list) and len(nbest_hyps) > 0 else []
            
            # CRITICAL FIX: Extract transcription directly from yseq to ensure accuracy
            # parse_hypothesis/add_results_to_json joins tokens with "" and replaces <space> with " "
            # But for SentencePiece, we need to handle ▁ prefix correctly
            transcription = ""
            if nbest_hyps and len(nbest_hyps) > 0:
                first_hyp = nbest_hyps[0]
                yseq = first_hyp.get('yseq', []) if isinstance(first_hyp, dict) else []
                if isinstance(yseq, torch.Tensor):
                    yseq_list = yseq.cpu().tolist()
                else:
                    yseq_list = list(yseq)
                
                # Extract tokens exactly like parse_hypothesis does (line 834-835)
                sos_eos_id = self.odim - 1
                token_texts = []
                for token_id in yseq_list[1:]:  # Skip SOS (first token) like parse_hypothesis does
                    if token_id == sos_eos_id:
                        break  # Stop at EOS
                    if token_id < len(self.token_list) and token_id >= 0:
                        token_text = self.token_list[token_id]
                        token_texts.append(token_text)
                
                # CRITICAL: Join tokens exactly like parse_hypothesis (line 841)
                # parse_hypothesis does: text = "".join(token_as_list).replace("<space>", " ")
                # For SentencePiece, ▁ indicates word boundary, so replace ▁ with space
                transcription = ''.join(token_texts)
                # Handle SentencePiece: ▁ prefix means word boundary (space)
                # Replace ▁ with space (except keep it if it's the entire token)
                transcription = transcription.replace('▁', ' ').replace('<space>', ' ')
                # Normalize multiple spaces to single space
                transcription = re.sub(r'\s+', ' ', transcription).strip()
                # Remove special tokens
                transcription = transcription.replace('<eos>', '').replace('<sos>', '').replace('<blank>', '').strip()
            else:
                # Fallback to add_results_to_json if yseq extraction fails
                transcription_raw = add_results_to_json(nbest_hyps, self.token_list) if nbest_hyps else ""
                transcription = transcription_raw.replace("▁", " ").replace("<space>", " ").strip()
                transcription = transcription.replace("<eos>", "").replace("<sos>", "").replace("<blank>", "").strip()
                transcription = re.sub(r'\s+', ' ', transcription).strip()
            
            # DEBUG: Log transcription and video info for debugging accuracy issues
            encoded_frame_count = enc_feats.shape[0] if hasattr(enc_feats, 'shape') else 0
            video_duration_seconds = encoded_frame_count / video_fps if video_fps > 0 else 0
            print(f"=" * 80)
            print(f"[MODEL] ===== TRANSCRIPTION DETAILED DEBUG =====")
            print(f"[MODEL] Video FPS: {video_fps}")
            print(f"[MODEL] Encoded frames: {encoded_frame_count}")
            print(f"[MODEL] Video duration: {video_duration_seconds:.2f} seconds")
            print(f"[MODEL] Transcription from beam search: {transcription}")
            print(f"[MODEL] Beam search parameters:")
            print(f"[MODEL]   - beam_size: {getattr(self.beam_search, 'beam_size', 'unknown')}")
            print(f"[MODEL]   - lm_weight: {getattr(self.beam_search, 'weights', {}).get('lm', 'unknown')}")
            print(f"=" * 80)
            
            # extract token IDs from best hypothesis for alignment
            # CRITICAL FIX: Extract token sequence matching the transcription (preserve order)
            if nbest_hyps and len(nbest_hyps) > 0:
                first_hyp = nbest_hyps[0]
                yseq = first_hyp.get('yseq', []) if isinstance(first_hyp, dict) else []
                # Convert to list if tensor
                if isinstance(yseq, torch.Tensor):
                    yseq_list = yseq.cpu().tolist()
                else:
                    yseq_list = list(yseq)
                
                # CRITICAL: parse_hypothesis skips first token (SOS), so we do the same
                # This ensures token_ids match the transcription text
                sos_eos_id = self.odim - 1
                # Skip SOS (first token) and remove EOS tokens, matching parse_hypothesis behavior
                token_ids = []
                for token_id in yseq_list[1:]:  # Skip first token (SOS) like parse_hypothesis does
                    if token_id != sos_eos_id and token_id < len(self.token_list) and token_id >= 0:
                        token_ids.append(token_id)
                
                # DEBUG: Log token extraction for debugging
                if len(token_ids) > 0:
                    token_texts = [self.token_list[tid] if tid < len(self.token_list) else f'<INVALID:{tid}>' for tid in token_ids[:10]]
                    print(f"[MODEL] Extracted {len(token_ids)} tokens, first 10: {token_texts}")
                else:
                    print(f"[MODEL] WARNING: No valid tokens extracted from yseq!")
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
                    
                    # DEBUG: Log alignment info
                    print(f"[MODEL] CTC alignment complete: {len(aligned_frames)} frames aligned, {len(token_ids)} tokens in sequence")
                    print(f"[MODEL] Encoded features shape: {enc_feats_for_align.shape}, Expected frames: {T}")
                    
                    # aligned_frames is a list of token IDs, one per frame
                    # CRITICAL FIX: Use transcription text directly (accurate) and map to timestamps
                    # Instead of reconstructing words from aligned_frames (which can have errors),
                    # we use the transcription text (correct from beam search) and map it to token segments
                    if len(aligned_frames) > 0 and len(token_ids) > 0:
                        # First pass: collect token segments from aligned_frames with frame ranges
                        token_segments = []
                        current_token = aligned_frames[0]
                        start_frame = 0
                        
                        for frame_idx, token_id in enumerate(aligned_frames):
                            token_changed = (token_id != current_token)
                            is_last_frame = (frame_idx == len(aligned_frames) - 1)
                            
                            if token_changed:
                                # Token changed - finish current segment
                                if current_token != 0 and current_token < len(self.token_list):  # skip blank tokens
                                    end_frame = frame_idx
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
                            
                            # Handle last frame
                            if is_last_frame:
                                if current_token != 0 and current_token < len(self.token_list):  # skip blank tokens
                                    end_frame = frame_idx + 1
                                    token_text = self.token_list[current_token]
                                    
                                    # Check if already added
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
                        
                        # CRITICAL FIX: Use transcription words directly (accurate from beam search)
                        # Map transcription words to frame ranges from aligned_frames
                        # This preserves accuracy while getting timestamps
                        transcription_words = [w.strip() for w in transcription.split() if w.strip()]
                        
                        # DEBUG: Log word counts for debugging
                        print(f"[MODEL] Transcription has {len(transcription_words)} words: {transcription_words}")
                        print(f"[MODEL] Token segments found: {len(token_segments)}, Aligned frames: {len(aligned_frames)}")
                        
                        # CRITICAL FIX: Group token_ids into words, ensuring proper word boundaries
                        # Match exactly how add_results_to_json creates the transcription
                        word_token_groups = []
                        current_word_token_ids = []
                        
                        for idx, token_id in enumerate(token_ids):
                            if token_id < len(self.token_list):
                                token_text = self.token_list[token_id]
                                # CRITICAL: Check if token has word boundary prefix (▁)
                                has_word_boundary = token_text.startswith('▁')
                                
                                if has_word_boundary:
                                    # Token with ▁ = start of new word
                                    if current_word_token_ids:
                                        # Finish previous word
                                        word_token_groups.append(current_word_token_ids)
                                        current_word_token_ids = [token_id]
                                    else:
                                        # First word or starting fresh
                                        current_word_token_ids = [token_id]
                                else:
                                    # Token without ▁ = continuation of current word
                                    if not current_word_token_ids:
                                        # Shouldn't happen (first token should have ▁), but handle it
                                        print(f"[MODEL] WARNING: Token {idx} without word boundary: {token_text}")
                                        current_word_token_ids = [token_id]
                                    else:
                                        # Continue current word
                                        current_word_token_ids.append(token_id)
                        
                        # CRITICAL: Add last word if exists
                        if current_word_token_ids:
                            word_token_groups.append(current_word_token_ids)
                        
                        # DEBUG: Verify word grouping matches transcription
                        if len(word_token_groups) != len(transcription_words):
                            print(f"[MODEL] CRITICAL ERROR: Word group count ({len(word_token_groups)}) != transcription word count ({len(transcription_words)})!")
                            print(f"[MODEL] Word groups: {len(word_token_groups)}")
                            print(f"[MODEL] Transcription words: {len(transcription_words)}")
                            # Try to reconstruct from tokens to see what's different
                            reconstructed = []
                            for group in word_token_groups:
                                word_parts = []
                                for tid in group:
                                    if tid < len(self.token_list):
                                        token_text = self.token_list[tid].replace('▁', '').strip()
                                        word_parts.append(token_text)
                                if word_parts:
                                    reconstructed.append(''.join(word_parts))
                            print(f"[MODEL] Reconstructed words from tokens: {reconstructed}")
                            print(f"[MODEL] Transcription words: {transcription_words}")
                        
                        # DEBUG: Log word grouping
                        print(f"[MODEL] Grouped into {len(word_token_groups)} word token groups (should match {len(transcription_words)} words)")
                        if len(word_token_groups) != len(transcription_words):
                            print(f"[MODEL] WARNING: Word count mismatch! Groups: {len(word_token_groups)}, Words: {len(transcription_words)}")
                        
                        # Create a map of token_id -> segments (for quick lookup)
                        token_segments_map = {}
                        for seg in token_segments:
                            token_id = seg['token_id']
                            if token_id not in token_segments_map:
                                token_segments_map[token_id] = []
                            token_segments_map[token_id].append(seg)
                        
                        # Map each word to its frame range from aligned_frames
                        # NOTE: Duplicate words are ALLOWED - video may legitimately say the same word multiple times
                        # e.g., "SAY WITH ME IRRITATE IRRITATE" - both "IRRITATE" should be included
                        for word_idx, word_token_ids in enumerate(word_token_groups):
                            if word_idx >= len(transcription_words):
                                break  # Safety check
                            
                            word_text = transcription_words[word_idx]  # Use transcription word (accurate)
                            
                            # Find frame ranges for all tokens in this word
                            word_start_frames = []
                            word_end_frames = []
                            
                            for token_id in word_token_ids:
                                if token_id in token_segments_map:
                                    # Get all segments for this token
                                    for seg in token_segments_map[token_id]:
                                        word_start_frames.append(seg['start_frame'])
                                        word_end_frames.append(seg['end_frame'])
                            
                            if word_start_frames and word_end_frames:
                                # Use min start and max end for the word
                                start_frame = min(word_start_frames)
                                end_frame = max(word_end_frames)
                                word_timestamps.append({
                                    'word': word_text,
                                    'start': start_frame / video_fps,
                                    'end': end_frame / video_fps
                                })
                                # Already tracked in processed_words above
                                
                                # DEBUG: Log first few words for debugging
                                if word_idx < 5:
                                    print(f"[MODEL] Word {word_idx+1}/{len(transcription_words)}: '{word_text}' → frames {start_frame}-{end_frame} ({start_frame/video_fps:.2f}s-{end_frame/video_fps:.2f}s)")
                            else:
                                # Fallback: distribute evenly if no alignment found
                                if word_timestamps:
                                    # Continue from last word's end
                                    last_end = word_timestamps[-1]['end']
                                    estimated_duration = (T / video_fps) / len(transcription_words) if len(transcription_words) > 0 else 0.5
                                    word_timestamps.append({
                                        'word': word_text,
                                        'start': last_end,
                                        'end': last_end + estimated_duration
                                    })
                                else:
                                    # First word
                                    estimated_duration = (T / video_fps) / len(transcription_words) if len(transcription_words) > 0 else 0.5
                                    word_timestamps.append({
                                        'word': word_text,
                                        'start': 0.0,
                                        'end': estimated_duration
                                    })
                        
                        # DEBUG: Log final word timestamps summary
                        print(f"=" * 80)
                        print(f"[MODEL] ===== WORD TIMESTAMPS SUMMARY =====")
                        print(f"[MODEL] Created {len(word_timestamps)} word timestamps from {len(transcription_words)} transcription words")
                        if len(word_timestamps) < len(transcription_words):
                            missing = len(transcription_words) - len(word_timestamps)
                            print(f"[MODEL] WARNING: {missing} words missing timestamps! Last word timestamps: {word_timestamps[-3:] if len(word_timestamps) >= 3 else word_timestamps}")
                            print(f"[MODEL] Missing words: {transcription_words[len(word_timestamps):]}")
                        
                        # DEBUG: Detailed time-based word detection for full video duration
                        print(f"[MODEL] ===== TIME-BASED WORD DETECTION (FULL DURATION) =====")
                        print(f"[MODEL] Video duration: {video_duration_seconds:.2f} seconds")
                        print(f"[MODEL] Word-by-word breakdown:")
                        for i, ws in enumerate(word_timestamps):
                            word = ws.get('word', '')
                            start = ws.get('start', 0.0)
                            end = ws.get('end', 0.0)
                            duration = end - start
                            # Create visual timeline
                            timeline_pos = int((start / video_duration_seconds) * 50) if video_duration_seconds > 0 else 0
                            timeline = ' ' * timeline_pos + '█' * max(1, int((duration / video_duration_seconds) * 50)) if video_duration_seconds > 0 else ''
                            print(f"[MODEL]   [{i+1:3d}] {start:6.2f}s - {end:6.2f}s ({duration:5.2f}s) | {word:20s} | {timeline}")
                        
                        # Create second-by-second breakdown
                        print(f"[MODEL] ===== SECOND-BY-SECOND BREAKDOWN =====")
                        for second in range(int(video_duration_seconds) + 1):
                            words_in_second = []
                            for ws in word_timestamps:
                                word_start = ws.get('start', 0.0)
                                word_end = ws.get('end', 0.0)
                                # Check if word overlaps with this second
                                if word_start < second + 1 and word_end > second:
                                    overlap_start = max(word_start, second)
                                    overlap_end = min(word_end, second + 1)
                                    overlap_duration = overlap_end - overlap_start
                                    words_in_second.append((ws.get('word', ''), overlap_start, overlap_end, overlap_duration))
                            
                            if words_in_second:
                                words_str = ', '.join([f"{w} ({s:.2f}s-{e:.2f}s)" for w, s, e, d in words_in_second])
                                print(f"[MODEL]   [{second:3d}s - {second+1:3d}s]: {words_str}")
                            else:
                                print(f"[MODEL]   [{second:3d}s - {second+1:3d}s]: (no words detected)")
                        
                        print(f"=" * 80)
                        
                        # Populate frame_alignments for debugging
                        for seg in token_segments:
                            token_text = seg['token_text']
                            clean_token = token_text.replace('<blank>', '').replace('<eos>', '').replace('<sos>', '').replace('▁', '').strip()
                            if clean_token and clean_token not in ['<blank>', '<eos>', '<sos>']:
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
            
            # CRITICAL: Free GPU memory before returning
            # Move intermediate tensors to CPU and delete them
            if torch.cuda.is_available():
                # Delete large tensors that are no longer needed
                if 'enc_feats' in locals():
                    del enc_feats
                if 'aligned_frames' in locals():
                    del aligned_frames
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
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
