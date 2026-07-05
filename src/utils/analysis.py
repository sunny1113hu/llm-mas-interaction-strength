from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data
