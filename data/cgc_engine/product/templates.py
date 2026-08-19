import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProductTemplate:
    name: str
    data: Dict[str, Any]
    path: str


def _templates_root() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    return project_root / "templates" / "m6"


def list_templates() -> List[str]:
    root = _templates_root()
    if not root.exists():
        return []
    return sorted([p.stem for p in root.glob("*.json")])


def load_template(name: str) -> ProductTemplate:
    root = _templates_root()
    path = root / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"template_not_found: {name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("name", "")).strip() != name:
        data["name"] = name
    return ProductTemplate(name=name, data=data, path=str(path))


def resolve_template(name: Optional[str]) -> ProductTemplate:
    nm = str(name or "").strip() or "ort_mnist_cpu"
    return load_template(nm)

