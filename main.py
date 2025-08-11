"""
XL-Share unified CLI runner

Provides a single entry point to run either the simple demo
or the full benchmark suite, and stores results in a
timestamped results directory.
"""

import argparse
import json
import os
import shutil
from datetime import datetime


def run_simple(output_dir: str) -> str:
    """Run the simplified experiments and store results in output_dir.

    Returns the path to the JSON results file.
    """
    from run_experiments_simple import main as simple_main

    os.makedirs(output_dir, exist_ok=True)
    results = simple_main()
    # Persist results in a stable filename inside output_dir
    outfile = os.path.join(output_dir, "simple_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # If the simple script also wrote a timestamped file at repo root,
    # move it into our output_dir for convenience.
    if isinstance(results, dict):
        meta = results.get("metadata", {})
        ts = meta.get("timestamp")
        if ts:
            legacy_name = f"simple_results_{ts}.json"
            if os.path.exists(legacy_name):
                try:
                    shutil.move(legacy_name, os.path.join(output_dir, legacy_name))
                except Exception:
                    pass
    return outfile


def run_full() -> str:
    """Run the full benchmark suite and return the produced results dir path.

    The full runner creates its own timestamped directory; we return that path.
    """
    from run_experiments import main as full_main

    results = full_main()
    # Detect directory from returned metadata
    try:
        ts = results.get("metadata", {}).get("timestamp")
        if ts:
            return f"results_{ts}"
    except Exception:
        pass
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XL-Share experiment runner")
    p.add_argument(
        "--mode",
        choices=["simple", "full", "calibrate"],
        default="simple",
        help="Which experiment suite to run",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: results_<timestamp>)",
    )
    p.add_argument(
        "--use-torch",
        action="store_true",
        help="Use PyTorch CUDA path for layer compute when available",
    )
    return p.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = args.out or f"results_{timestamp}"

    if args.mode == "calibrate":
        from xlshare.hardware_calibration import run_calibration
        out_dir = base_out
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "calibration.json")
        res = run_calibration(out)
        print(f"Calibration written to {out}")
    elif args.mode == "simple":
        out_dir = base_out
        path = run_simple(out_dir)
        print(f"Simple results saved: {path}")
        print(f"Artifacts directory: {out_dir}")
    else:
        # Full suite manages its own results_<timestamp> directory
        # Set env hint for torch usage (picked up inside benchmarks/engine)
        if args.use_torch:
            os.environ["XL_USE_TORCH"] = "1"
        res_dir = run_full()
        if res_dir:
            print(f"Full results directory: {res_dir}")
        else:
            print("Full suite completed; see console output for details.")


if __name__ == "__main__":
    main()
