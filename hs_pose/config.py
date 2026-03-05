import json

from hs_pose.constants import (
    CONFIG_PATH,
    DEFAULT_CONFIDENCE,
    DEFAULT_RTSP_TRANSPORT,
    DEFAULT_RTSP_URL,
)


def load_config() -> dict:
    default_config = {
        "rtsp_url": DEFAULT_RTSP_URL,
        "confidence": DEFAULT_CONFIDENCE,
        "transport": DEFAULT_RTSP_TRANSPORT,
        "game": {
            "pixel_count": 120,
            "charge_rate": 1.0,
            "active_decay_rate": 0.15,
            "idle_decay_rate": 0.35,
            "idle_drain_enabled": True,
            "takeover_decay_enabled": True,
            "tick_hz": 30,
        },
        "sacn": {
            "enabled": False,
            "receiver_ip": "",
            "universe": 1,
            "start_address": 1,
            "test_mode_enabled": False,
            "test_palette": "Manual RGB",
            "test_r": 255,
            "test_g": 64,
            "test_b": 64,
        },
    }
    if not CONFIG_PATH.exists():
        return default_config

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return default_config

    if not isinstance(data, dict):
        return default_config

    confidence = data.get("confidence", DEFAULT_CONFIDENCE)
    if not isinstance(confidence, (int, float)):
        confidence = DEFAULT_CONFIDENCE
    confidence = min(max(float(confidence), 0.0), 1.0)
    transport = str(data.get("transport", DEFAULT_RTSP_TRANSPORT)).lower()
    if transport not in {"auto", "tcp", "udp"}:
        transport = DEFAULT_RTSP_TRANSPORT

    game_data = data.get("game", {})
    if not isinstance(game_data, dict):
        game_data = {}

    def _to_int(value, fallback: int, minimum: int) -> int:
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return fallback

    def _to_float(value, fallback: float, minimum: float) -> float:
        try:
            return max(minimum, float(value))
        except (TypeError, ValueError):
            return fallback

    pixel_count = _to_int(
        game_data.get("pixel_count", default_config["game"]["pixel_count"]),
        default_config["game"]["pixel_count"],
        1,
    )
    charge_rate = _to_float(
        game_data.get("charge_rate", default_config["game"]["charge_rate"]),
        default_config["game"]["charge_rate"],
        0.0,
    )
    active_decay_rate = _to_float(
        game_data.get("active_decay_rate", default_config["game"]["active_decay_rate"]),
        default_config["game"]["active_decay_rate"],
        0.0,
    )
    idle_decay_rate = _to_float(
        game_data.get("idle_decay_rate", default_config["game"]["idle_decay_rate"]),
        default_config["game"]["idle_decay_rate"],
        0.0,
    )
    tick_hz = _to_int(
        game_data.get("tick_hz", default_config["game"]["tick_hz"]),
        default_config["game"]["tick_hz"],
        1,
    )
    idle_drain_enabled = bool(
        game_data.get("idle_drain_enabled", default_config["game"]["idle_drain_enabled"])
    )
    takeover_decay_enabled = bool(
        game_data.get(
            "takeover_decay_enabled",
            default_config["game"]["takeover_decay_enabled"],
        )
    )
    sacn_data = data.get("sacn", {})
    if not isinstance(sacn_data, dict):
        sacn_data = {}
    sacn_enabled = bool(sacn_data.get("enabled", default_config["sacn"]["enabled"]))
    receiver_ip = str(sacn_data.get("receiver_ip", default_config["sacn"]["receiver_ip"]))
    universe = _to_int(
        sacn_data.get("universe", default_config["sacn"]["universe"]),
        default_config["sacn"]["universe"],
        1,
    )
    start_address = _to_int(
        sacn_data.get("start_address", default_config["sacn"]["start_address"]),
        default_config["sacn"]["start_address"],
        1,
    )
    test_mode_enabled = bool(
        sacn_data.get("test_mode_enabled", default_config["sacn"]["test_mode_enabled"])
    )
    test_palette = str(sacn_data.get("test_palette", default_config["sacn"]["test_palette"]))
    test_r = _to_int(
        sacn_data.get("test_r", default_config["sacn"]["test_r"]),
        default_config["sacn"]["test_r"],
        0,
    )
    test_g = _to_int(
        sacn_data.get("test_g", default_config["sacn"]["test_g"]),
        default_config["sacn"]["test_g"],
        0,
    )
    test_b = _to_int(
        sacn_data.get("test_b", default_config["sacn"]["test_b"]),
        default_config["sacn"]["test_b"],
        0,
    )

    return {
        "rtsp_url": data.get("rtsp_url") or DEFAULT_RTSP_URL,
        "confidence": confidence,
        "transport": transport,
        "game": {
            "pixel_count": pixel_count,
            "charge_rate": charge_rate,
            "active_decay_rate": active_decay_rate,
            "idle_decay_rate": idle_decay_rate,
            "idle_drain_enabled": idle_drain_enabled,
            "takeover_decay_enabled": takeover_decay_enabled,
            "tick_hz": tick_hz,
        },
        "sacn": {
            "enabled": sacn_enabled,
            "receiver_ip": receiver_ip,
            "universe": min(63999, universe),
            "start_address": min(512, start_address),
            "test_mode_enabled": test_mode_enabled,
            "test_palette": test_palette,
            "test_r": min(255, test_r),
            "test_g": min(255, test_g),
            "test_b": min(255, test_b),
        },
    }


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
