import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
import random


class ScoreLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, input_channels, depth, width, grad_checkpointing=False, alpha=1, noise_channels=64, train_temperature=1):
        super(ScoreLoss, self).__init__()
        self.noise_channels = noise_channels
        self.net = SimpleMLPAdaLN(
            input_channels=input_channels,
            model_channels=width,
            out_channels=target_channels,
            noise_channels=noise_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )
        self.alpha = alpha
        self.train_temperature = train_temperature

    def forward(self, target, z, mask=None):
        noise = torch.rand((z.shape[0] * 2, self.noise_channels), dtype=z.dtype, device=z.device) - 0.5
        z = z.repeat(2, 1)
        sample = self.net(z, noise, 1)
        sample_1, sample_2 = sample.chunk(2, dim = 0)
        score = self.energy_score(sample_1, sample_2, target)
        loss = - score
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def energy_distance(self, x_1, x_2):
        return torch.pow(torch.linalg.norm(x_1 - x_2, ord=2, dim=-1), self.alpha)

    def energy_score(self, sample_1, sample_2, target, additional_targets = None):
        distance_1 = self.energy_distance(sample_1, target)
        distance_2 = self.energy_distance(sample_2, target)
        variance = self.energy_distance(sample_1, sample_2)
        score = self.train_temperature * variance - distance_1 - distance_2
        return score

    def sample(self, z, infer_temperature=1.0, cfg=1.0):
        if cfg != 1.0:
            z_1, z_2 = z.chunk(2, dim=0)
            z = z_1 * cfg + (1 - cfg) * z_2
        noise = torch.rand((z.shape[0], self.noise_channels), dtype=z.dtype, device=z.device) - 0.5
        return self.net(z, noise, infer_temperature).float()

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y, infer_temperature):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        if not self.training:
            shift_mlp = infer_temperature * shift_mlp
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Score Loss.
    :param input_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param noise_channels: channels in the noise.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        input_channels,
        model_channels,
        out_channels,
        noise_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.grad_checkpointing = grad_checkpointing

        self.noise_embed = nn.Linear(noise_channels, model_channels)
        self.input_proj = nn.Linear(input_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, z, noise, infer_temperature):
        z = self.input_proj(z)
        noise = self.noise_embed(noise)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                z = checkpoint(block, z, noise, infer_temperature)
        else:
            for block in self.res_blocks:
                z = block(z, noise, infer_temperature)

        return self.final_layer(z, noise)
