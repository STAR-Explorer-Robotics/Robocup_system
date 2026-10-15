"""Command-line entry point for inspection-run."""

from __future__ import annotations

import json
import sys

from rescue_inspection.cli_commands.dispatch import dispatch_command
from rescue_inspection.cli_commands.parser import build_parser


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        payload = dispatch_command(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
