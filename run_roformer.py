"""Standalone driver for audio-separator (Roformer / MDXC) separation.

Mirrors ``run_demucs.py``: a thin subprocess entry point that ``server.py``
shells out to, so the heavy ``audio_separator`` + ``onnxruntime`` import stays
out of the main FastAPI process until a Roformer job actually runs. This keeps
the default Demucs/Whisper/CREPE path unaffected and lets the two separators
use independent CUDA allocations (one process at a time, gated by the server's
MAX_CONCURRENT).

Demucs models go through ``run_demucs.py``; Roformer/MDXC checkpoints
(e.g. ``BS-Roformer-SW.ckpt``, a 6-stem vocals/drums/bass/guitar/piano/other
model) go through here.

Also supports a ``--download-only`` mode that loads the requested checkpoint
(triggering its download) and then exits without separating — used by
server.py's startup warmup when the default model is a Roformer model.

Output files land in the output directory named::

    <input-stem>_(<stem-label>)_<model>.flac

which server.py maps back to stem names.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="audio-separator checkpoint filename (e.g. BS-Roformer-SW.ckpt)")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("--model-dir", default=None,
                        help="directory where checkpoints are cached/downloaded")
    parser.add_argument("-d", "--device", default="",
                        help="cpu or cuda — informational; CUDA visibility is "
                             "controlled by the caller via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--download-only", action="store_true",
                        help="download/load the model then exit (warmup)")
    parser.add_argument("input", nargs="?", help="input audio file")
    args = parser.parse_args()

    # Imported here (not at module top) so a missing audio-separator install
    # only breaks Roformer jobs, never the Demucs/Whisper/CREPE endpoints.
    from audio_separator.separator import Separator

    sep_kwargs = {"output_dir": args.output, "output_format": "FLAC"}
    if args.model_dir:
        sep_kwargs["model_file_dir"] = args.model_dir

    separator = Separator(**sep_kwargs)
    print(f"[run_roformer] Loading model {args.model}...", flush=True)
    separator.load_model(model_filename=args.model)

    if args.download_only:
        print(f"[run_roformer] {args.model} ready.", flush=True)
        return 0

    if not args.input:
        print("[run_roformer] no input file provided", file=sys.stderr, flush=True)
        return 2

    outputs = separator.separate(args.input)
    # Echo produced filenames on a stable, easy-to-parse prefix so the
    # caller does not have to guess audio-separator's naming scheme.
    for name in outputs or []:
        print(f"OUTPUT\t{name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
