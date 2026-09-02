#!/usr/bin/env python
"""Streaming ASR server for nvidia/parakeet-unified-en-0.6b (NeMo runtime).

v2: true buffered streaming, following NeMo's own
examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py
(the model card's recommended path): the encoder runs over a rolling
[left | chunk | right] window with chunked-limited attention, and a stateful
greedy label-looping RNNT decoder carries continuity across chunks. Emitted
labels are FINAL — no LocalAgreement, no eviction heuristics, no text merging.

Protocol (client: _RemoteUnifiedModel in parakeet_asr.py):
  stdin :  <u32 LE byte length> <float32 mono 16 kHz PCM>   -> push audio
           length 0xFFFFFFFF                                 -> flush (end of stream)
  stdout:  one JSON line per request with the WHOLE stream so far:
             {"tokens": [...], "timestamps": [...]}   (client diffs by count)
           tokens are SentencePiece pieces with '▁' already turned into a
           leading space, so "".join(tokens) is the text.
  startup: {"ready": true} once the model is loaded and warmed.

Context preset via env: UNIFIED_LEFT_SECS (5.6), UNIFIED_CHUNK_SECS (1.04),
UNIFIED_RIGHT_SECS (1.04) -> theoretical latency = chunk + right.
"""
import json
import os
import struct
import sys

import numpy as np

SAMPLE_RATE = 16000
FLUSH = 0xFFFFFFFF


def main() -> None:
    out = sys.stdout
    sys.stdout = sys.stderr  # keep the protocol channel clean

    import logging

    logging.getLogger("nemo_logger").setLevel(logging.ERROR)

    import torch
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf, open_dict

    torch.set_grad_enabled(False)

    try:
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
        from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
        from nemo.collections.asr.parts.utils.streaming_utils import (
            ContextSize,
            StreamingBatchedAudioBuffer,
        )

        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-unified-en-0.6b"
        )
        if model.cfg.get("validation_ds", None) is None:
            with open_dict(model.cfg):
                model.cfg.validation_ds = OmegaConf.create({})
        dec_cfg = RNNTDecodingConfig(fused_batch_size=-1)
        dec_cfg.strategy = "greedy_batch"
        model.change_decoding_strategy(dec_cfg)
        model = model.to("cuda").eval()
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        computer = model.decoding.decoding.decoding_computer

        feature_stride_sec = model.cfg.preprocessor["window_stride"]
        feats_per_sec = 1.0 / feature_stride_sec
        sub = model.encoder.subsampling_factor
        f2a = int(SAMPLE_RATE * feature_stride_sec)
        f2a -= f2a % sub  # make divisible by subsampling factor
        enc_frame2audio = f2a * sub

        L = float(os.environ.get("UNIFIED_LEFT_SECS", "5.6"))
        C = float(os.environ.get("UNIFIED_CHUNK_SECS", "1.04"))
        R = float(os.environ.get("UNIFIED_RIGHT_SECS", "1.04"))
        cef = ContextSize(
            left=int(L * feats_per_sec / sub),
            chunk=int(C * feats_per_sec / sub),
            right=int(R * feats_per_sec / sub),
        )
        cs = ContextSize(
            left=cef.left * sub * f2a,
            chunk=cef.chunk * sub * f2a,
            right=cef.right * sub * f2a,
        )
        if (
            model.cfg.encoder.get("att_context_style", "") == "chunked_limited_with_rc"
        ):
            model.encoder.set_default_att_context_size(
                att_context_size=[cef.left, cef.chunk, cef.right]
            )
        print(
            f"[unified server] contexts sec L/C/R = {cs.left/SAMPLE_RATE:.2f}/"
            f"{cs.chunk/SAMPLE_RATE:.2f}/{cs.right/SAMPLE_RATE:.2f} "
            f"(theoretical latency {(cs.chunk+cs.right)/SAMPLE_RATE:.2f}s)",
            flush=True,
        )

        buffer = StreamingBatchedAudioBuffer(
            batch_size=1, context_samples=cs, dtype=torch.float32, device="cuda"
        )
        state = None
        cur_hyps = None
        pending = np.zeros(0, dtype=np.float32)
        need = cs.chunk + cs.right  # first window needs chunk+right before decoding
        decoded_sec = 0.0
        all_tokens: list[str] = []
        all_times: list[float] = []
        known = 0

        def extract_new() -> None:
            nonlocal known
            hyp = batched_hyps_to_hypotheses(cur_hyps, batch_size=1)[0]
            ids = hyp.y_sequence.tolist() if hasattr(hyp.y_sequence, "tolist") else list(hyp.y_sequence)
            if len(ids) <= known:
                return
            pieces = model.tokenizer.ids_to_tokens(ids[known:])
            for p in pieces:
                all_tokens.append(p.replace("▁", " "))
                all_times.append(decoded_sec)
            known = len(ids)

        def step(is_last: bool = False) -> None:
            nonlocal pending, need, state, cur_hyps, decoded_sec
            while len(pending) >= need or (is_last and len(pending) > 0):
                take = min(need, len(pending)) if is_last else need
                seg = torch.from_numpy(pending[:take]).unsqueeze(0).to("cuda")
                pending = pending[take:]
                last = is_last and len(pending) == 0
                lens = torch.tensor([take], device="cuda")
                buffer.add_audio_batch_(
                    seg,
                    audio_lengths=lens,
                    is_last_chunk=last,
                    is_last_chunk_batch=torch.tensor([last], device="cuda"),
                )
                enc, enc_len = model(
                    input_signal=buffer.samples,
                    input_signal_length=buffer.context_size_batch.total(),
                )
                enc = enc.transpose(1, 2)
                ecs = buffer.context_size.subsample(factor=enc_frame2audio)
                ecb = buffer.context_size_batch.subsample(factor=enc_frame2audio)
                enc = enc[:, ecs.left:]
                if last:
                    dlen = enc_len - ecb.left
                else:
                    dlen = ecb.chunk
                hyps, state = computer(x=enc, out_len=dlen, prev_batched_state=state)
                if cur_hyps is None:
                    cur_hyps = hyps
                else:
                    cur_hyps.merge_(hyps)
                extract_new()
                decoded_sec += take / SAMPLE_RATE
                need = cs.chunk
                if last:
                    break

        # Warm up (compile kernels) with a second of silence through a throwaway state.
        _ = model(
            input_signal=torch.zeros(1, cs.chunk + cs.right, device="cuda"),
            input_signal_length=torch.tensor([cs.chunk + cs.right], device="cuda"),
        )
    except Exception as e:
        print(json.dumps({"ready": False, "error": f"{type(e).__name__}: {e}"}),
              file=out, flush=True)
        return

    print(json.dumps({"ready": True}), file=out, flush=True)

    stdin = sys.stdin.buffer
    while True:
        hdr = stdin.read(4)
        if len(hdr) < 4:
            return
        (n,) = struct.unpack("<I", hdr)
        try:
            if n == FLUSH:
                step(is_last=True)
                print(json.dumps({"tokens": all_tokens, "timestamps": all_times, "eos": True}),
                      file=out, flush=True)
                return
            buf = b""
            while len(buf) < n:
                chunk = stdin.read(n - len(buf))
                if not chunk:
                    return
                buf += chunk
            pending = np.concatenate([pending, np.frombuffer(buf, dtype=np.float32)])
            step()
            print(json.dumps({"tokens": all_tokens, "timestamps": all_times}),
                  file=out, flush=True)
        except Exception as e:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), file=out, flush=True)


if __name__ == "__main__":
    main()
