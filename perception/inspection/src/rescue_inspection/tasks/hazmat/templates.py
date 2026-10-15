"""Hazmat templates manager."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from rescue_inspection.project_paths import resolve_project_path
from rescue_inspection.tasks.hazmat.label_registry import LabelRegistry
from rescue_inspection.tasks.hazmat.template_images import read_hazmat_template_bgr


class HazmatTemplateManager:
    """Loads and provides reference templates for Hazmat signs."""

    def __init__(
        self,
        template_dir: Path | str = "samples/Hazmat2020",
        target_size: int = 256,
        label_registry: LabelRegistry | None = None,
    ) -> None:
        self.registry = label_registry
        self.template_dir = resolve_project_path(label_registry.template_root if label_registry else template_dir)
        self.target_size = target_size
        self.templates: dict[str, np.ndarray] = {}
        self.labels: list[str] = []
        self._load_templates()

    def _load_templates(self) -> None:
        if not self.template_dir.exists():
            logging.warning("Template directory %s does not exist. Hazmat classifier will fail.", self.template_dir)
            return

        if self.registry is not None:
            entries = self.registry.labels
            image_paths = [(entry.key, self.template_dir / entry.filename) for entry in entries]
        else:
            image_paths = [
                (img_path.stem.replace("placard-", ""), img_path)
                for img_path in sorted(self.template_dir.glob("*.png"))
            ]

        for name, img_path in image_paths:
            img = read_hazmat_template_bgr(img_path)
            if img is None:
                continue
            resized = cv2.resize(img, (self.target_size, self.target_size))
            self.templates[name] = resized
            self.labels.append(name)

        logging.info("Loaded %d hazmat templates from %s", len(self.templates), self.template_dir)
