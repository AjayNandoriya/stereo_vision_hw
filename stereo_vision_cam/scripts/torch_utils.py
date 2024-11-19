import torch
import math
import cv2

class IntrinsicDistortion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k1 = torch.nn.Parameter(torch.randn(()))
        self.k2 = torch.nn.Parameter(torch.randn(()))
        self.p1 = torch.nn.Parameter(torch.randn(()))
        self.p2 = torch.nn.Parameter(torch.randn(()))
        self.k3 = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        r2 = r*r
        r4 = r2*r2
        r6 = r2*r4
        1 + self.k1*r2 + self.k2*r2 + self.k3*r6
        


def test_intrinsic_distortion():
    cv2.