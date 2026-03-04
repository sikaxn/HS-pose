from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
MODEL_PATH = BASE_DIR / "ISAhs_weight" / "weights" / "best.pt"
POSE_MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"
DEFAULT_RTSP_URL = "rtsp://10.0.0.181:554/live/av0"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
APPSINK_MAX_BUFFERS = 1
FRAME_WAIT_TIMEOUT_MS = 200
STREAM_CONNECT_TIMEOUT_MS = 3000
