"""Wrapper to run demucs with a patched save_audio that uses soundfile instead of torchaudio.save.

torchaudio >= 2.11 requires torchcodec for saving, which has shared library issues.
This wrapper patches demucs.audio.save_audio to use soundfile directly.

Also supports a ``--download-only`` mode that loads the requested model
(triggering its CDN download via torch.hub) and then exits without
running separation. Used by server.py's startup warmup.
"""

import sys
import soundfile as sf
import torch


def patched_save_audio(wav, path, samplerate=44100, bitrate=320, clip="rescale",
                       bits_per_sample=16, as_float=False, **kwargs):
    """Save audio tensor to WAV using soundfile."""
    path = str(path)
    if not path.endswith('.wav'):
        path = path + '.wav'
    # wav shape: (channels, samples) — soundfile expects (samples, channels)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if clip == "rescale":
        mx = wav.abs().max()
        if mx > 0:
            wav = wav / max(mx, 1e-8)
    elif clip == "clamp":
        wav = wav.clamp(-1, 1)
    data = wav.T.cpu().numpy()
    subtype = 'FLOAT' if as_float else 'PCM_16'
    sf.write(path, data, samplerate, subtype=subtype)


# Patch demucs before importing its main
import demucs.audio
demucs.audio.save_audio = patched_save_audio


def _download_only() -> int:
    """Load the requested model (downloading via torch.hub if needed),
    then exit. Args mirror the subset of demucs.separate flags we use:
    -n MODEL, -d DEVICE.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("-n", "--name", default="htdemucs_ft")
    parser.add_argument("-d", "--device", default="cpu")
    args, _unknown = parser.parse_known_args()

    # demucs.pretrained.get_model() is the stable cross-version API on
    # demucs >= 4.0.0; it triggers the torch.hub weights download to
    # the standard cache and returns the loaded model object. Don't
    # use demucs.api.Separator — that module is missing in some
    # demucs 4.0.x wheels.
    from demucs.pretrained import get_model
    print(f"[run_demucs] Pre-downloading {args.name} weights...", flush=True)
    get_model(args.name)
    print(f"[run_demucs] {args.name} ready.", flush=True)
    return 0


if "--download-only" in sys.argv:
    sys.exit(_download_only())

# Now run demucs main
from demucs.separate import main
main()
