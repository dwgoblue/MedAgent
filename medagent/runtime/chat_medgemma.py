#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from medagent.runtime.tools.medgemma import LocalMedGemmaClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Terminal chat with local MedGemma")
    p.add_argument("--system", default="You are a concise medical decision-support assistant.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.temperature is not None:
        os.environ["MEDAGENT_MEDGEMMA_TEMPERATURE"] = str(args.temperature)

    os.environ.setdefault("MEDAGENT_USE_MEDGEMMA", "1")
    client = LocalMedGemmaClient()

    history: list[dict[str, str]] = [{"role": "system", "content": args.system}]

    print("MedGemma terminal chat. Type /exit to quit, /reset to clear history.")
    print("Model ref:", os.getenv("MEDAGENT_MEDGEMMA_MODEL_DIR") or os.getenv("MEDAGENT_MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it"))

    while True:
        try:
            user = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return 0

        if not user:
            continue
        if user in {"/exit", "exit", "quit"}:
            return 0
        if user == "/reset":
            history = [{"role": "system", "content": args.system}]
            print("assistant> history reset")
            continue

        history.append({"role": "user", "content": user})
        try:
            reply = client.chat(history, max_new_tokens=args.max_new_tokens)
        except Exception as exc:
            print(f"assistant> [error] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        if not reply:
            reply = "[empty response]"
        print("assistant>", reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
