from __future__ import annotations

from typing import List, Sequence, Tuple


TEST_PALETTES = (
    "Manual RGB",
    "Rainbow",
    "Color Wipe",
    "Theater Chase",
    "Police",
)


def build_test_pixels(
    pixel_count: int,
    palette: str,
    manual_rgb: Sequence[int],
    elapsed_seconds: float,
) -> List[Tuple[int, int, int]]:
    count = max(1, int(pixel_count))
    t = max(0.0, float(elapsed_seconds))

    if palette == "Rainbow":
        return [_hsv_to_rgb((idx / count + t * 0.1) % 1.0, 1.0, 1.0) for idx in range(count)]

    if palette == "Color Wipe":
        color = tuple(int(max(0, min(255, channel))) for channel in manual_rgb[:3])
        off = (0, 0, 0)
        lit = int((t * 15) % (count + 1))
        return [color if idx < lit else off for idx in range(count)]

    if palette == "Theater Chase":
        color = tuple(int(max(0, min(255, channel))) for channel in manual_rgb[:3])
        off = (0, 0, 0)
        phase = int(t * 8) % 3
        return [color if (idx + phase) % 3 == 0 else off for idx in range(count)]

    if palette == "Police":
        blink = int(t * 3) % 2
        first = (255, 0, 0) if blink == 0 else (0, 0, 255)
        second = (0, 0, 255) if blink == 0 else (255, 0, 0)
        split = count // 2
        return [first if idx < split else second for idx in range(count)]

    color = tuple(int(max(0, min(255, channel))) for channel in manual_rgb[:3])
    return [color] * count


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> Tuple[int, int, int]:
    h = (hue % 1.0) * 6.0
    c = value * saturation
    x = c * (1 - abs(h % 2 - 1))
    m = value - c

    if 0 <= h < 1:
        r, g, b = c, x, 0
    elif 1 <= h < 2:
        r, g, b = x, c, 0
    elif 2 <= h < 3:
        r, g, b = 0, c, x
    elif 3 <= h < 4:
        r, g, b = 0, x, c
    elif 4 <= h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (
        int((r + m) * 255),
        int((g + m) * 255),
        int((b + m) * 255),
    )
