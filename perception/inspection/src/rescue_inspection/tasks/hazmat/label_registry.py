"""Hazmat label manifest loading and lookup helpers."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from rescue_inspection.project_paths import resolve_project_path


UNKNOWN_LABEL = "UNKNOWN"


class HazmatLabelEntry(BaseModel):
    """One label definition from the manifest."""

    id: int
    key: str
    filename: str
    display_name: str
    official_name: str
    aliases: list[str] = Field(default_factory=list)


class HazmatLabelManifest(BaseModel):
    """On-disk manifest schema for Hazmat labels."""

    version: int = 1
    template_root: str = "samples/Hazmat2020"
    default_label_set: str = "hazmat2020_full_15"
    label_sets: dict[str, list[str]] = Field(default_factory=dict)
    labels: list[HazmatLabelEntry] = Field(default_factory=list)


def default_manifest_path() -> Path:
    """Return the default Hazmat labels manifest path."""

    return resolve_project_path("data/manifests/hazmat_labels.yaml")


def build_default_manifest(template_root: str = "samples/Hazmat2020") -> HazmatLabelManifest:
    """Create the default 15-class manifest."""

    labels = [
        HazmatLabelEntry(
            id=index,
            key=key,
            filename=f"placard-{key}.png",
            display_name=display_name,
            official_name=official_name,
            aliases=list(aliases),
        )
        for index, key, display_name, official_name, aliases in [
            (0, "1.1-explosives", "1.1 Explosives", "Explosives 1.1", ("explosives-1.1", "explosives")),
            (1, "1.5-blasting-agents", "1.5 Blasting Agents", "Blasting Agents 1.5", ("blasting-agents-1.5", "blasting-agents")),
            (2, "2-flammable-gas", "2 Flammable Gas", "Flammable Gas", ("flammable-gas", "class-2-flammable-gas")),
            (3, "2-non-flammable-gas", "2 Non-Flammable Gas", "Non-Flammable Gas", ("non-flammable-gas", "class-2-non-flammable-gas")),
            (4, "2-oxygen", "2 Oxygen", "Oxygen", ("oxygen", "oxidizing-gas")),
            (5, "3-fuel-oil", "3 Fuel Oil", "Fuel Oil", ("fuel-oil", "class-3-fuel-oil")),
            (6, "4-dangerous-when-wet", "4 Dangerous When Wet", "Dangerous When Wet", ("dangerous-when-wet", "class-4-dangerous-when-wet")),
            (7, "4-flammable-solid", "4 Flammable Solid", "Flammable Solid", ("flammable-solid", "class-4-flammable-solid")),
            (8, "4-spontaneously-combustible", "4 Spontaneously Combustible", "Spontaneously Combustible", ("spontaneously-combustible", "class-4-spontaneously-combustible")),
            (9, "5.1-oxidizer", "5.1 Oxidizer", "Oxidizer", ("oxidizer", "class-5.1-oxidizer")),
            (10, "5.2-organic-peroxide", "5.2 Organic Peroxide", "Organic Peroxide", ("organic-peroxide", "class-5.2-organic-peroxide")),
            (11, "6-inhalation-hazard", "6 Inhalation Hazard", "Inhalation Hazard", ("inhalation-hazard", "class-6-inhalation-hazard")),
            (12, "6-poison", "6 Poison", "Poison", ("poison", "class-6-poison")),
            (13, "7-radioactive", "7 Radioactive", "Radioactive", ("radioactive", "class-7-radioactive")),
            (14, "8-corrosive", "8 Corrosive", "Corrosive", ("corrosive", "class-8-corrosive")),
        ]
    ]
    full_set = [label.key for label in labels]
    return HazmatLabelManifest(
        version=1,
        template_root=template_root,
        default_label_set="hazmat2020_full_15",
        label_sets={
            "hazmat2020_full_15": full_set,
            "competition_active": [],
        },
        labels=labels,
    )


def write_default_manifest(path: Path | str | None = None, template_root: str = "samples/Hazmat2020") -> Path:
    """Write the default 15-class manifest and return its path."""

    registry = LabelRegistry(build_default_manifest(template_root=template_root))
    return registry.save(path)


class LabelRegistry:
    """Lookup wrapper around the Hazmat labels manifest."""

    def __init__(self, manifest: HazmatLabelManifest, *, manifest_path: Path | None = None) -> None:
        self.manifest = manifest
        self.manifest_path = manifest_path
        self._by_key = {label.key: label for label in manifest.labels}
        self._by_id = {label.id: label for label in manifest.labels}

    @classmethod
    def from_file(cls, path: Path | str | None = None) -> "LabelRegistry":
        """Load a registry from a YAML file."""

        manifest_path = resolve_project_path(path or default_manifest_path())
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(HazmatLabelManifest.model_validate(payload), manifest_path=manifest_path)

    def save(self, path: Path | str | None = None) -> Path:
        """Write the current manifest to disk."""

        target_path = resolve_project_path(path or self.manifest_path or default_manifest_path())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.manifest.model_dump(mode="python"), handle, sort_keys=False, allow_unicode=False)
        return target_path

    @property
    def template_root(self) -> Path:
        """Resolved template directory."""

        return resolve_project_path(self.manifest.template_root)

    @property
    def labels(self) -> list[HazmatLabelEntry]:
        """All registered labels."""

        return list(self.manifest.labels)

    def full_labels(self) -> list[HazmatLabelEntry]:
        """Return labels from the manifest default set."""

        return [self._by_key[key] for key in self.manifest.label_sets.get(self.manifest.default_label_set, []) if key in self._by_key]

    def active_label_keys(self, label_set: str | None = None) -> list[str]:
        """Return the active label keys for the requested set."""

        selected = label_set or self.manifest.default_label_set
        keys = list(self.manifest.label_sets.get(selected, []))
        if not keys and selected == "competition_active":
            return []
        if not keys and selected != self.manifest.default_label_set:
            return list(self.manifest.label_sets.get(self.manifest.default_label_set, []))
        return keys

    def active_labels(self, label_set: str | None = None) -> list[HazmatLabelEntry]:
        """Return active label entries."""

        return [self._by_key[key] for key in self.active_label_keys(label_set) if key in self._by_key]

    def key_to_id(self, key: str) -> int:
        """Map a label key to its numeric id."""

        return self._by_key[key].id

    def id_to_key(self, label_id: int) -> str:
        """Map a numeric id back to the label key."""

        return self._by_id[label_id].key

    def key_to_display_name(self, key: str) -> str:
        """Return the display name for a label key."""

        return self._by_key[key].display_name

    def key_to_official_name(self, key: str) -> str:
        """Return the official name for a label key."""

        return self._by_key[key].official_name

    def get_label(self, key: str) -> HazmatLabelEntry:
        """Return a label entry by key."""

        return self._by_key[key]

    def is_active(self, key: str, label_set: str | None = None) -> bool:
        """Return whether the label is active in the selected set."""

        return key in set(self.active_label_keys(label_set))

    def map_prediction(self, key: str, label_set: str | None = None) -> str:
        """Map inactive or unknown labels to the runtime UNKNOWN token."""

        if key not in self._by_key:
            return UNKNOWN_LABEL
        return key if self.is_active(key, label_set) else UNKNOWN_LABEL
