from __future__ import annotations

from typing import Iterable, List

import cv2
import numpy as np
import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs, output) -> None:  # type: ignore[no-untyped-def]
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:  # type: ignore[no-untyped-def]
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(input_tensor)
        score = logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Khong lay duoc activation/gradient cho Grad-CAM")

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)

        cam = cam - cam.min()
        max_val = cam.max()
        if max_val > 0:
            cam = cam / max_val

        return cam.detach().cpu().numpy().astype(np.float32)


def find_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        layer4 = getattr(model, "layer4")
        if len(layer4) > 0:
            return layer4[-1]

    if hasattr(model, "features"):
        features = getattr(model, "features")
        if len(features) > 0:
            return features[-1]

    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module

    raise ValueError("Khong tim thay Conv2d layer de tinh Grad-CAM")


def combine_cams(cams: Iterable[np.ndarray], weights: Iterable[float] | None = None) -> np.ndarray:
    cam_list = [cam for cam in cams]
    if not cam_list:
        raise ValueError("Danh sach CAM rong")

    if weights is None:
        weights_arr = np.ones(len(cam_list), dtype=np.float32)
    else:
        weights_arr = np.asarray(list(weights), dtype=np.float32)
        if len(weights_arr) != len(cam_list):
            raise ValueError("So luong weights khong khop so CAM")

    merged = np.zeros_like(cam_list[0], dtype=np.float32)
    for cam, w in zip(cam_list, weights_arr):
        merged = np.maximum(merged, cam * max(float(w), 0.0))

    merged = merged - merged.min()
    max_val = merged.max()
    if max_val > 0:
        merged = merged / max_val
    return merged.astype(np.float32)


def overlay_cam(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0.0)


def pick_cam_classes(target: str, fire_prob: float, smoke_prob: float, min_prob: float) -> List[int]:
    target = target.lower()
    if target == "fire":
        return [0]
    if target == "smoke":
        return [1]

    classes: List[int] = []
    if fire_prob >= min_prob:
        classes.append(0)
    if smoke_prob >= min_prob:
        classes.append(1)

    if not classes:
        classes = [0, 1]
    return classes
