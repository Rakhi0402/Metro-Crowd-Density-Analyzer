"""
config.py
---------
Central configuration for the crowd-counting application.
Edit these values to tune detection behaviour without touching
the core logic.
"""


class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    # YOLOv8 model weights.  Pre-trained options (downloaded automatically):
    #   "yolov8n.pt" – nano  (fastest, least accurate)  ← bad for crowds
    #   "yolov8s.pt" – small (good balance for crowds)
    #   "yolov8m.pt" – medium (better accuracy, slower)
    #   "yolov8l.pt" – large
    #   "yolov8x.pt" – extra-large (slowest, most accurate)
    MODEL_PATH: str = "yolov8s.pt"

    # ── Source ─────────────────────────────────────────────────────────────
    # 0  → default webcam
    # 1, 2, … → other camera indices
    # "/path/to/video.mp4" → video file
    SOURCE = 0  # type: int | str

    # ── Detection ──────────────────────────────────────────────────────────
    # Minimum confidence for a detection to be kept (0–1).
    # For dense crowds: lower = catches more occluded/partial people.
    # Recommended range: 0.20–0.35 for crowded scenes.
    CONFIDENCE_THRESHOLD: float = 0.25

    # Non-maximum suppression IoU threshold (0–1).
    # For dense crowds: LOWER this so tightly-packed people are NOT merged.
    # Higher value = more boxes suppressed (bad for crowds).
    # Recommended range: 0.30–0.40 for crowded scenes.
    NMS_IOU_THRESHOLD: float = 0.35

    # COCO class index for "person"
    PERSON_CLASS_ID: int = 0

    # ── Tracking ───────────────────────────────────────────────────────────
    # Tracker config bundled with Ultralytics.
    # "bytetrack.yaml"  – ByteTrack  (fast, robust, recommended)
    # "botsort.yaml"    – BoT-SORT   (more accurate, heavier)
    TRACKER: str = "bytetrack.yaml"

    # ── Optical Flow Algorithm (IEEE 7275777) ──────────────────────────────
    OF_BLUR_KERNEL: tuple[int, int] = (5, 5)
    OF_DIFF_THRESHOLD: int = 25
    # FAST threshold (lower = catches more features, higher = more selective)
    OF_FAST_THRESHOLD: int = 10
    # Max points to track (increase for denser 'crowd cloud' effect)
    OF_MAX_POINTS: int = 600
    # Minimum movement (pixels) to count as a moving person (filters out jitter)
    OF_MIN_MOTION: float = 1.0
    # Maximum movement (pixels) between frames beyond which it's likely noise/error
    OF_MAX_MOTION: float = 50.0
    # Spatial Grid
    OF_GRID_ROWS: int = 4
    OF_GRID_COLS: int = 3
    OF_GRID_HIGH_THRESH: int = 15
    OF_GRID_MED_THRESH: int = 5

    # Heuristic: roughly how many motion vectors correspond to one person
    # (Tweak this based on your camera distance and crowd density)
    OF_VECTORS_PER_PERSON: float = 30.0  # Dense flow needs a higher ratio

    # ── Crowd / Risk Thresholds ──────────────────────────────────────────
    CROWD_MED_THRESHOLD: int = 20
    CROWD_HIGH_THRESHOLD: int = 50
    ALERT_THRESHOLD: int = 15
    RISK_MODERATE_THRESHOLD: float = 12.0
    RISK_HIGH_THRESHOLD: float = 18.0
    RISK_CRITICAL_THRESHOLD: float = 25.0

    # ── Dense Farneback Parameters ──────────────────────────────────────
    OF_DENSE_PYRAMID_SCALE: float = 0.5
    OF_DENSE_LEVELS: int = 3
    OF_DENSE_WINSIZE: int = 15
    OF_DENSE_ITERATIONS: int = 3
    OF_DENSE_POLY_N: int = 5
    OF_DENSE_POLY_SIGMA: float = 1.2
    OF_DENSE_SAMPLE_STEP: int = 15  # Sample every N pixels for visualization

    WAIT_DELAY: int = 1


    # ── Display ────────────────────────────────────────────────────────────
    # Window title shown by OpenCV
    WINDOW_TITLE: str = "Crowd Counter – YOLOv8 + ByteTrack"

    # Bounding-box colour (BGR)
    BOX_COLOR: tuple = (0, 255, 0)

    # Thickness of bounding-box lines (pixels)
    BOX_THICKNESS: int = 2

    # Colour of the track-ID label text (BGR)
    ID_COLOR: tuple = (255, 255, 255)

    # Colour for the HUD overlay text (BGR)
    HUD_COLOR: tuple = (0, 255, 255)

    # Font scale for HUD text
    HUD_FONT_SCALE: float = 0.9

    # HUD text thickness (pixels)
    HUD_FONT_THICKNESS: int = 2

    # ── Processing ─────────────────────────────────────────────────────────
    # Frame width to resize to before inference.
    # None = keep original resolution (recommended for most cases).
    # Set to e.g. 1280 to LIMIT max width (will only downscale, never upscale).
    FRAME_WIDTH: int | None = None

    # Key (ord value) to press to quit
    QUIT_KEY: int = ord("q")

    # ── Forecasting ────────────────────────────────────────────────────────
    # Path to trained LSTM model for crowd forecasting
    FORECAST_MODEL: str = "forecasting/lstm_model.pth"
