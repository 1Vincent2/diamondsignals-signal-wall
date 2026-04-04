from __future__ import annotations

from typing import Iterable


def _to_float_list(values: Iterable) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            if v is None:
                continue
            out.append(float(v))
        except Exception:
            continue
    return out


def build_sparkline(
    values: Iterable,
    width: int = 120,
    height: int = 34,
    pad_x: int = 2,
    pad_y: int = 2,
) -> dict:
    series = _to_float_list(values)

    if not series:
        return {
            "points": f"0,{height//2} {width},{height//2}",
            "trend_direction": "flat",
            "trend_glow": False,
            "trend_delta": 0.0,
            "sample_size": 0,
            "has_history": False,
        }

    if len(series) == 1:
        y = height / 2
        return {
            "points": f"{pad_x},{y:.1f} {width - pad_x},{y:.1f}",
            "trend_direction": "flat",
            "trend_glow": False,
            "trend_delta": 0.0,
            "sample_size": 1,
            "has_history": False,
        }

    vmin = min(series)
    vmax = max(series)
    vrange = vmax - vmin

    usable_width = max(width - 2 * pad_x, 1)
    usable_height = max(height - 2 * pad_y, 1)

    points: list[str] = []
    n = len(series)

    for i, value in enumerate(series):
        x = pad_x + (usable_width * i / (n - 1))
        if vrange == 0:
            y = pad_y + usable_height / 2
        else:
            y = pad_y + (usable_height * (1 - ((value - vmin) / vrange)))
        points.append(f"{x:.1f},{y:.1f}")

    first = series[0]
    last = series[-1]
    delta = last - first

    if abs(delta) < 1e-9:
        direction = "flat"
    elif delta > 0:
        direction = "up"
    else:
        direction = "down"

    if n >= 3 and direction == "up":
        glow = True
    else:
        glow = False

    return {
        "points": " ".join(points),
        "trend_direction": direction,
        "trend_glow": glow,
        "trend_delta": round(delta, 4),
        "sample_size": n,
        "has_history": True,
    }