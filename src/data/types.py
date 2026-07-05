from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Problem:
    problem_id: str
    context: str
    question: str
    answer: str
    evidence_ids: List[str]
    depth: Optional[int] = None
    triples: Optional[Dict[str, str]] = None
    rules: Optional[Dict[str, str]] = None
    local_contexts: Optional[Dict[int, str]] = None
    metadata: Optional[Dict[str, object]] = None
