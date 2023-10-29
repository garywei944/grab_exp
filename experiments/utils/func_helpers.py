from torch import nn, Tensor


def make_func_params(
    model: nn.Module,
) -> tuple[dict[str, nn.Parameter], dict[str, Tensor]]:
    # https://pytorch.org/docs/master/func.migrating.html#functorch-make-functional
    # https://pytorch.org/docs/stable/generated/torch.func.functional_call.html
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    params = {k: v for k, v in params.items() if v.requires_grad}
    for v in params.values():
        v.requires_grad_(False)

    return params, buffers
