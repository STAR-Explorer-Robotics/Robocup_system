"""Label manifest command handlers."""

from __future__ import annotations

import argparse


def run_build_hazmat_labels_command(args: argparse.Namespace) -> dict[str, object]:
    """Write the default Hazmat labels manifest."""

    from rescue_inspection.tasks.hazmat.label_registry import write_default_manifest

    path = write_default_manifest(path=args.output, template_root=args.template_root)
    return {"labels_file": str(path)}

