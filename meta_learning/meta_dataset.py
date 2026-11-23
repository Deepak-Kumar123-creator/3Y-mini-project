import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetaDatasetManager:
    """Stores real Task 1 experiments as structured meta-examples."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read historical meta-dataset %s: %s",
                self.path,
                exc,
            )
            return []

    def add(self, experiment: dict[str, Any]) -> list[dict[str, Any]]:
        experiments = self.load()
        fingerprint = experiment.get("dataset_fingerprint")
        experiments = [
            item for item in experiments
            if (
                item.get("dataset_fingerprint"),
                item.get("task_type"),
                item.get("target"),
            )
            != (
                fingerprint,
                experiment.get("task_type"),
                experiment.get("target"),
            )
        ]
        experiments.append(experiment)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(experiments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return experiments
