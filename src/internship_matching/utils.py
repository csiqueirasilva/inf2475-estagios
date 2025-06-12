# internship_matching/utils.py
import functools
import torch

def get_device():
    """Return 'cuda' if available, else 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42):
    import random; import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        import functools
import warnings

def deprecated(reason: str = ""):
    """
    Decorator to mark functions as deprecated.
    :param reason: Optional explanation (e.g. alternative function name).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__}(). {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator