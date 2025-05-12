import torch
from ..utils import get_device

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sanitize_input_jobs():
    str = get_device()
    print(f"Device: {str}")