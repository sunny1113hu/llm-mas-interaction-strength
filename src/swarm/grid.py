from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Grid:
    size: int

    def to_pos(self, agent_id: int) -> Tuple[int, int]:
        return divmod(agent_id, self.size)

    def to_id(self, row: int, col: int) -> int:
        return (row % self.size) * self.size + (col % self.size)

    def neighbors(self, agent_id: int) -> List[int]:
        row, col = self.to_pos(agent_id)
        return [
            self.to_id(row - 1, col),
            self.to_id(row + 1, col),
            self.to_id(row, col - 1),
            self.to_id(row, col + 1),
        ]
