from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


CLOTH_ORDER = (
    "green-scorates",
    "yellow-mandela",
    "red-teresa",
    "blue-malala",
)

CLOTH_COLORS: Dict[str, Tuple[int, int, int]] = {
    "green-scorates": (0, 220, 90),
    "yellow-mandela": (255, 215, 0),
    "red-teresa": (245, 64, 64),
    "blue-malala": (48, 140, 255),
}


@dataclass
class GameParams:
    pixel_count: int = 120
    charge_rate: float = 1.0
    active_decay_rate: float = 0.15
    idle_decay_rate: float = 0.35
    idle_drain_enabled: bool = True
    takeover_decay_enabled: bool = True


class EnergyGameEngine:
    def __init__(self, params: GameParams | None = None) -> None:
        self.params = params or GameParams()
        self._energy = {cloth: 0.0 for cloth in CLOTH_ORDER}

    def set_params(self, params: GameParams) -> None:
        self.params = params

    def reset(self) -> None:
        for cloth in self._energy:
            self._energy[cloth] = 0.0

    def update(
        self,
        waving_counts: Dict[str, int],
        delta_seconds: float,
        shirt_counts: Dict[str, int] | None = None,
    ) -> List[Tuple[int, int, int]]:
        dt = max(0.0, float(delta_seconds))
        if dt <= 0.0:
            return self._build_pixels()

        active_total = 0
        active_cloth_count = 0
        active_map = {}
        for cloth in CLOTH_ORDER:
            active = max(0, int(waving_counts.get(cloth, 0)))
            active_map[cloth] = active
            active_total += active
            if active > 0:
                active_cloth_count += 1
            self._energy[cloth] += self.params.charge_rate * active * dt

        # Takeover decay only applies when multiple cloth colors are active together.
        if active_cloth_count >= 2 and self.params.takeover_decay_enabled:
            for cloth in CLOTH_ORDER:
                if active_map[cloth] == 0:
                    self._energy[cloth] = max(
                        0.0,
                        self._energy[cloth] - self.params.active_decay_rate * dt,
                    )

        shirt_total = 0
        if shirt_counts:
            for cloth in CLOTH_ORDER:
                shirt_total += max(0, int(shirt_counts.get(cloth, 0)))

        if active_total == 0 and shirt_total == 0 and self.params.idle_drain_enabled:
            present_cloths = [cloth for cloth in CLOTH_ORDER if self._energy[cloth] > 1e-9]
            if present_cloths:
                drain_each = (self.params.idle_decay_rate * dt) / len(present_cloths)
                for cloth in present_cloths:
                    self._energy[cloth] = max(0.0, self._energy[cloth] - drain_each)

        self._cap_total_energy()
        return self._build_pixels()

    def get_energy(self) -> Dict[str, float]:
        return dict(self._energy)

    def _build_pixels(self) -> List[Tuple[int, int, int]]:
        pixel_count = max(1, int(self.params.pixel_count))
        total_energy = sum(self._energy.values())
        if total_energy <= 1e-9:
            return [(0, 0, 0)] * pixel_count

        lit_pixels = min(pixel_count, int(total_energy))
        if lit_pixels <= 0:
            return [(0, 0, 0)] * pixel_count

        raw_counts = []
        remainders = []
        used = 0
        for cloth in CLOTH_ORDER:
            raw = (self._energy[cloth] / total_energy) * lit_pixels
            count = int(raw)
            raw_counts.append([cloth, count])
            remainders.append((raw - count, cloth))
            used += count

        remaining = lit_pixels - used
        for _fraction, cloth in sorted(remainders, key=lambda item: item[0], reverse=True):
            if remaining <= 0:
                break
            for row in raw_counts:
                if row[0] == cloth:
                    row[1] += 1
                    remaining -= 1
                    break

        pixels: List[Tuple[int, int, int]] = []
        for cloth, count in raw_counts:
            pixels.extend([CLOTH_COLORS[cloth]] * count)

        if len(pixels) < pixel_count:
            pixels.extend([(0, 0, 0)] * (pixel_count - len(pixels)))
        return pixels[:pixel_count]

    def _cap_total_energy(self) -> None:
        pixel_count = max(1, int(self.params.pixel_count))
        total_energy = sum(self._energy.values())
        if total_energy <= pixel_count:
            return

        scale = pixel_count / total_energy
        for cloth in CLOTH_ORDER:
            self._energy[cloth] *= scale
