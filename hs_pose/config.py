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

    return {
        "rtsp_url": data.get("rtsp_url") or DEFAULT_RTSP_URL,
        "confidence": confidence,
        "transport": transport,
    }


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
