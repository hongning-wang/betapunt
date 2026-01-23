from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class AlphaHoldemNetConfig:
    trunk_channels: int = 128
    num_blocks: int = 6
    fc_dim: int = 2048
    # Fixed by the JAX training code
    card_chans: int = 7
    act_chans: int = 30
    nb_actions: int = 13
    suits: int = 4
    ranks: int = 13

    @property
    def flat_dim(self) -> int:
        # Conv trunk outputs (B, 4, 13, trunk_channels) in NHWC before flatten.
        return self.suits * self.ranks * self.trunk_channels


class ResidualBlockTorch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv0(x))
        y = self.conv1(y)
        return torch.relu(x + y)


class AlphaHoldemNetTorch(nn.Module):
    """PyTorch equivalent of `train.py::AlphaHoldemNet`.

    Inputs are expected in NHWC to match the JAX code:
      - card_tensor: (B, 4, 13, 7)
      - action_tensor: (B, 4, 13, 30)
    """

    def __init__(self, cfg: AlphaHoldemNetConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings over the last dim of NHWC tensors.
        self.card_emb = nn.Linear(cfg.card_chans, 128, bias=True)
        self.act_emb = nn.Linear(cfg.act_chans, 128, bias=True)

        # Conv trunk operates on NCHW.
        self.conv_in = nn.Conv2d(128, cfg.trunk_channels, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.ModuleList([ResidualBlockTorch(cfg.trunk_channels) for _ in range(cfg.num_blocks)])

        # Fully-connected head
        self.fc1 = nn.Linear(cfg.flat_dim, cfg.fc_dim, bias=True)
        self.fc2 = nn.Linear(cfg.fc_dim, cfg.fc_dim, bias=True)
        self.policy_head = nn.Linear(cfg.fc_dim, cfg.nb_actions, bias=True)
        self.value_head = nn.Linear(cfg.fc_dim, 1, bias=True)

    def forward(self, card_tensor: torch.Tensor, action_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NHWC -> embed over last dim
        x = self.card_emb(card_tensor) + self.act_emb(action_tensor)  # (B,4,13,128)

        # NHWC -> NCHW for conv
        x = x.permute(0, 3, 1, 2).contiguous()  # (B,128,4,13)

        x = torch.relu(self.conv_in(x))
        for b in self.blocks:
            x = b(x)

        # NCHW -> NHWC to match flatten order in JAX
        x = x.permute(0, 2, 3, 1).contiguous()  # (B,4,13,C)
        x = x.reshape(x.shape[0], -1)  # (B, 4*13*C)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        logits = self.policy_head(x)  # (B,13)
        value = self.value_head(x).squeeze(-1)  # (B,)
        return logits, value


def _as_numpy(x: Any) -> np.ndarray:
    # Works for JAX arrays, numpy arrays, and python scalars.
    return np.array(x, copy=True)


def load_flax_params_into_torch(model: AlphaHoldemNetTorch, flax_params: Dict[str, Any]) -> None:
    """Load Flax/JAX params (from `train.py`) into the PyTorch module in-place."""

    def put_linear(prefix: str, torch_linear: nn.Linear) -> None:
        k = _as_numpy(flax_params[prefix]["kernel"])  # (in, out)
        b = _as_numpy(flax_params[prefix]["bias"])  # (out,)
        torch_linear.weight.data.copy_(torch.from_numpy(k.T).to(torch_linear.weight.dtype))
        torch_linear.bias.data.copy_(torch.from_numpy(b).to(torch_linear.bias.dtype))

    def put_conv(prefix: str, torch_conv: nn.Conv2d) -> None:
        k = _as_numpy(flax_params[prefix]["kernel"])  # (kh, kw, in, out)
        b = _as_numpy(flax_params[prefix]["bias"])  # (out,)
        w = np.transpose(k, (3, 2, 0, 1))  # (out, in, kh, kw)
        torch_conv.weight.data.copy_(torch.from_numpy(w).to(torch_conv.weight.dtype))
        torch_conv.bias.data.copy_(torch.from_numpy(b).to(torch_conv.bias.dtype))

    # Embeddings
    put_linear("Dense_0", model.card_emb)
    put_linear("Dense_1", model.act_emb)

    # Trunk
    put_conv("Conv_0", model.conv_in)
    for i, blk in enumerate(model.blocks):
        p = flax_params[f"ResidualBlock_{i}"]
        k0 = _as_numpy(p["Conv_0"]["kernel"])
        b0 = _as_numpy(p["Conv_0"]["bias"])
        k1 = _as_numpy(p["Conv_1"]["kernel"])
        b1 = _as_numpy(p["Conv_1"]["bias"])
        blk.conv0.weight.data.copy_(torch.from_numpy(np.transpose(k0, (3, 2, 0, 1))).to(blk.conv0.weight.dtype))
        blk.conv0.bias.data.copy_(torch.from_numpy(b0).to(blk.conv0.bias.dtype))
        blk.conv1.weight.data.copy_(torch.from_numpy(np.transpose(k1, (3, 2, 0, 1))).to(blk.conv1.weight.dtype))
        blk.conv1.bias.data.copy_(torch.from_numpy(b1).to(blk.conv1.bias.dtype))

    # Heads
    put_linear("Dense_2", model.fc1)
    put_linear("Dense_3", model.fc2)
    put_linear("Dense_4", model.policy_head)
    put_linear("Dense_5", model.value_head)


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    # Dynamic quantization only affects Linear layers; Conv2d stays float.
    # Ensure a quantization engine is selected (e.g. macOS often needs qnnpack).
    if torch.backends.quantized.engine in (None, "", "none"):
        supported = getattr(torch.backends.quantized, "supported_engines", [])
        if "qnnpack" in supported:
            torch.backends.quantized.engine = "qnnpack"
        elif "fbgemm" in supported:
            torch.backends.quantized.engine = "fbgemm"
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def build_model_from_config_dict(cfg_dict: Dict[str, Any]) -> AlphaHoldemNetTorch:
    cfg = AlphaHoldemNetConfig(**cfg_dict)
    return AlphaHoldemNetTorch(cfg)


def load_exported_torch_model(path: str, *, quantized: bool = False, device: str = "cpu") -> nn.Module:
    payload = torch.load(path, map_location="cpu")
    cfg_dict = payload["config"]

    model: nn.Module = build_model_from_config_dict(cfg_dict)
    if quantized:
        model = quantize_dynamic_int8(model)

    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model.to(device)

