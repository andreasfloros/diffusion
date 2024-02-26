import torch as th
from typing import Optional


def save_state(checkpoint_path: str,
               model: Optional[th.nn.Module] = None,
               ema: Optional[th.optim.swa_utils.AveragedModel] = None,
               optimizer: Optional[th.optim.Optimizer] = None) -> None:
    """
    Save the model, ema and optimizer state dicts.
    """

    state_dict = {}
    if model is not None:
        state_dict["model_state_dict"] = model.state_dict()
    if ema is not None:
        state_dict["ema_state_dict"] = ema.state_dict()
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    th.save(state_dict, checkpoint_path)


def load_state(checkpoint_path: str,
               model: Optional[th.nn.Module] = None,
               ema: Optional[th.optim.swa_utils.AveragedModel] = None,
               optimizer: Optional[th.optim.Optimizer] = None) -> None:
    """
    Load the model, ema and optimizer state dicts.
    """

    checkpoint = th.load(checkpoint_path, map_location="cpu")
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if ema is not None:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def count_parameters(model: th.nn.Module) -> int:
    """
    Count the number of parameters in a model.
    """

    return sum(p.numel() for p in model.parameters())


@th.inference_mode()
def ddim(diffuser: th.nn.Module,
         init: th.Tensor,
         condition: Optional[th.Tensor] = None,
         steps: Optional[int] = None,
         eta: float = 1.,
         clamp_min: float = -th.inf,
         clamp_max: float = th.inf) -> th.Tensor:
    """
    Diffuse a tensor under a DDIM schedule.
    Defaults to DDPM for steps=None, eta=1.

    Args:
        diffuser: contains the non-blind denoiser (epsilon prediction).
            Should take ([x, condition], t) as input if conditional and (x, t) otherwise.
        init: the initial tensor.
        condition: the condition tensor. None if the denoiser is not conditional.
        steps: the number of steps to diffuse.
        eta: the eta parameter.
        clamp_min: the minimum value of the clamp.
        clamp_max: the maximum value of the clamp.
    """

    x = init
    if not steps:
        steps = int(diffuser.T)
    times = th.linspace(int(diffuser.T) - 1, 0, steps, dtype=int, device=x.device)

    for index in range(len(times)):
        t = times[index]
        sqrt_alpha_cumprod = diffuser.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = diffuser.sqrt_one_minus_alphas_cumprod[t]

        inpt = th.cat((x, condition), 1) if condition is not None else x
        eps_pred = diffuser.epsilon(inpt, t)
        x0_pred = (x - sqrt_one_minus_alpha_cumprod * eps_pred) / sqrt_alpha_cumprod
        x0_pred.clamp_(clamp_min, clamp_max)
        eps_pred = (x - sqrt_alpha_cumprod * x0_pred) / sqrt_one_minus_alpha_cumprod

        if index == len(times) - 1:
            return x0_pred

        prev_t = times[index + 1]
        alpha_cumprod = diffuser.alphas_cumprod[t]
        alpha_cumprod_prev = diffuser.alphas_cumprod[prev_t]
        sqrt_alpha_cumprod_prev = diffuser.sqrt_alphas_cumprod[prev_t]
        sqrt_one_minus_alpha_cumprod_prev = diffuser.sqrt_one_minus_alphas_cumprod[prev_t]

        std = eta * (sqrt_one_minus_alpha_cumprod_prev / sqrt_one_minus_alpha_cumprod) * th.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)  # noqa: E501
        x = sqrt_alpha_cumprod_prev * x0_pred + th.sqrt(1 - alpha_cumprod_prev - std ** 2) * eps_pred + std * th.randn_like(x)  # noqa: E501
