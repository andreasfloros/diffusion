import improved_diffusion
import torch as th
from typing import Any, Dict, Optional, Tuple
import utils


class Diffuser(th.nn.Module):
    """
    VP diffuser module.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 T: int,
                 linear: bool,
                 unet_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.unet = improved_diffusion.UNetModel(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 channel_mult=[1,] + unet_cfg["channel_mults"],
                                                 model_channels=unet_cfg["base_channels"],
                                                 num_res_blocks=unet_cfg["num_res_attn_blocks"],
                                                 attention_resolutions=[2 ** i for i, is_attn in enumerate(unet_cfg["is_attn"]) if is_attn],  # noqa: E501
                                                 dropout=unet_cfg["dropout"],
                                                 num_heads=unet_cfg["num_heads"],
                                                 use_scale_shift_norm=unet_cfg["use_scale_shift_norm"],
                                                 )
        if linear:
            betas = th.linspace(0.1 / T, 20 / T, T, dtype=th.float64)
        else:
            s = 0.008
            steps = th.linspace(0., T, T + 1, dtype=th.float64)
            ft = th.cos(((steps / T + s) / (1 + s)) * th.pi * 0.5) ** 2
            betas = th.clip(1 - ft[1:] / ft[:T], 0., 0.999)

        sqrt_betas = th.sqrt(betas)
        alphas = 1 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0)
        one_minus_alphas_cumprod = 1 - alphas_cumprod
        sqrt_alphas = th.sqrt(alphas)

        sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = th.sqrt(one_minus_alphas_cumprod)

        self.register_buffer("betas", betas.to(th.float32))
        self.register_buffer("sqrt_betas", sqrt_betas.to(th.float32))
        self.register_buffer("alphas", alphas.to(th.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(th.float32))
        self.register_buffer("one_minus_alphas_cumprod", one_minus_alphas_cumprod.to(th.float32))
        self.register_buffer("sqrt_alphas", sqrt_alphas.to(th.float32))
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod.to(th.float32))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod.to(th.float32))

        T = th.tensor(T, dtype=th.float32).unsqueeze_(0)
        self.register_buffer("T", T)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        DDP
        """

        t = self.randint(batch_size=x.shape[0], device=x.device)
        xt, eps = self.noise(x, t)
        return th.nn.functional.mse_loss(self.epsilon(xt, t), eps)

    def noise(self, x: th.Tensor, t: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Noising from 0 to t.
        """

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        eps = th.randn_like(x)
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * eps, eps

    def randint(self, batch_size: int, device: th.device) -> th.Tensor:
        """
        Sample a random time step.
        """

        return th.randint(low=0, high=len(self.betas), size=(batch_size, 1, 1, 1), device=device)

    def epsilon(self, x: th.Tensor, t: th.Tensor) -> th.Tensor:
        return self.unet(x, t * 1000. / len(self.betas))

    @th.inference_mode()
    def sample(self,
               init: th.Tensor,
               steps: Optional[int] = None,
               eta: float = 1.,
               clamp_min: float = -1.,
               clamp_max: float = 1.) -> th.Tensor:
        """
        Sample from the model.
        """

        return utils.ddim(diffuser=self, init=init, steps=steps, eta=eta, clamp_min=clamp_min, clamp_max=clamp_max)
