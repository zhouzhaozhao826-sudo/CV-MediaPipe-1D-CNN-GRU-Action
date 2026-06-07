from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class SquatRepCounter:
    alpha: float = 0.25
    margin_ratio: float = 0.06
    history_size: int = 90
    peak_window: int = 5
    rep_count: int = 0
    state: str = "standing"
    smoothed_value: float | None = None
    min_value: float = 1.0
    max_value: float = 0.0
    history: deque[float] = field(default_factory=lambda: deque(maxlen=90))

    def update(self, hip_height: float) -> int:
        if self.smoothed_value is None:
            self.smoothed_value = hip_height
        else:
            self.smoothed_value = self.alpha * hip_height + (1.0 - self.alpha) * self.smoothed_value

        value = float(self.smoothed_value)
        self.history.append(value)
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

        dynamic_range = self.max_value - self.min_value
        if dynamic_range < 0.02:
            return self.rep_count

        midpoint = (self.max_value + self.min_value) / 2.0
        margin = dynamic_range * self.margin_ratio
        bottom_threshold = midpoint + margin
        top_threshold = midpoint - margin

        if self.state == "standing" and value > bottom_threshold:
            self.state = "descending"
        elif self.state == "descending" and self._is_at_peak(value):
            self.state = "bottom"
        elif self.state == "bottom" and value < top_threshold:
            self.rep_count += 1
            self.state = "standing"

        return self.rep_count

    def _is_at_peak(self, current: float) -> bool:
        if len(self.history) < self.peak_window:
            return False
        recent = list(self.history)[-self.peak_window:]
        return current <= max(recent)
