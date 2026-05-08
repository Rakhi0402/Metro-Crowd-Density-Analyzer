from config import Config


def classify(count, cfg: Config | None = None):
    """
    Classify crowd density level based on total count.

    Parameters
    ----------
    count : int
        Total people count.
    cfg : Config | None
        Central configuration values for crowd thresholds.

    Returns:
        "LOW", "MEDIUM", or "HIGH"
    """
    if cfg is None:
        cfg = Config()

    if count < cfg.CROWD_MED_THRESHOLD:
        return "LOW"
    elif count < cfg.CROWD_HIGH_THRESHOLD:
        return "MEDIUM"
    return "HIGH"