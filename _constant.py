from pathlib import Path


DEFAULT_PLATFORM_STOP_TIME = 60
DEFAULT_EXIT_WALKING_TIME = 60
TRAIN_CAPACITY_FACTOR = 0.15
DEMAND_RATE = 0.668 #


def manual_input_path(filename):
    """Locate a seed specification in the current or legacy data layout."""
    nested_path = Path("data/manual_input_data") / filename
    if nested_path.exists():
        return nested_path
    return Path("data") / filename
