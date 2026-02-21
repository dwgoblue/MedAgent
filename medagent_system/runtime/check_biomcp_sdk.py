#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from medagent_system.runtime.tools.retrieval.biomcp_sdk_client import (
    biomcp_sdk_available,
    query_biomcp_sdk,
)


def main() -> int:
    p = argparse.ArgumentParser(description="BioMCP SDK smoke check")
    p.add_argument("--intent", default="GENERAL_LITERATURE_SUPPORT")
    p.add_argument("--query", default="BRAF melanoma")
    p.add_argument("--max-results", type=int, default=2)
    args = p.parse_args()

    out = {
        "sdk_available": biomcp_sdk_available(),
        "intent": args.intent,
        "query": args.query,
        "max_results": args.max_results,
        "results": [],
    }
    if out["sdk_available"]:
        rows, debug = query_biomcp_sdk(args.intent, args.query, max_results=args.max_results)
        out["results"] = [
            {
                "citation_id": r.citation_id,
                "source": r.source,
                "summary": r.summary[:200],
                "metadata": r.metadata,
            }
            for r in rows
        ]
        out["debug"] = {
            "ok": debug.ok,
            "result_count": debug.result_count,
            "attempted_calls": debug.attempted_calls,
            "errors": debug.errors,
        }

    print(json.dumps(out, indent=2))
    return 0 if out["sdk_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
