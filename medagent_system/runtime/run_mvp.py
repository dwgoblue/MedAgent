from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct script execution from repo root without package install.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medagent_system.runtime.harness.mvp_harness import run_mvp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedAgent runtime on a CPB JSON file")
    parser.add_argument(
        "--cpb",
        type=str,
        default="medagent_system/runtime/examples/sample_cpb.json",
        help="Path to CPB JSON input",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path (prints to stdout when omitted)",
    )
    args = parser.parse_args()

    cpb_path = Path(args.cpb)
    output = run_mvp(cpb_path)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved output to: {out}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
