"""Gesture recognition model and factory methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 63
HIDDEN_DIM = 32
WINDOW_SIZE = 32
SAMPLE_FPS = 12

CLASSES = [
    "No Gesture",
    "Swipe Left",
    "Swipe Right",
    "Open Twice",
    "Grab"
]


class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (padding, 0)))


class GestureCNN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM,
                 hidden_dim=HIDDEN_DIM, num_classes=len(CLASSES)):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.temporal = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, 5),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3, dilation=2),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3, dilation=4),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3, dilation=8),
            nn.ReLU()
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.projection(x)
        x = self.temporal(x.transpose(1, 2))
        return self.classifier(x[:, :, -1])

def get_model(**kwargs):
    return GestureCNN(**kwargs)

def load_model(weights_path=None, device="cpu"):
    model = get_model()
    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
    return model.to(device).eval()
