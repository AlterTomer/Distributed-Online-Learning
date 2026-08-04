"""Functional evaluation, flattening, and Jacobian products.

The one place ``torch.func`` is used. Everything here works on a plain
``nn.Module`` **template** whose own parameter values are never read -- the
values always arrive as an argument.

**Why Jacobian products rather than a Jacobian.** The filter needs
$\\bm H_{v,t} = \\partial\\bm h_{v,t}/\\partial\\bm\\theta^{\\mathsf T} \\in
\\mathbb R^{q\\times p}$. At $q=10$ and $p=2908$ that matrix is small, but the
recipe has to survive $p\\sim10^5$, where materialising it is $10^6$ entries per
sample. Every quantity the filter actually wants -- the information increment
$\\bm H^{\\mathsf T}\\bm\\Lambda\\bm H$, the innovation term
$\\bm H^{\\mathsf T}\\bm\\nu$ -- is a product against $\\bm H$, so exposing
``vjp`` and ``jvp`` and never forming $\\bm H$ is the scalable route
(research note §5.3). :func:`jacobian` exists for tests and small models, and
says so.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call, jvp, vjp

from dekf_bench.models.base import ParamDict, ParamGroup


class FunctionalError(ValueError):
    """Raised when a flat vector or tangent does not match the model."""


def parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Parameter names in the module's own order.

    Insertion order, not sorted: it is stable for a given module definition and
    it keeps a layer's weight and bias adjacent, so ``param_groups`` spans are
    contiguous. Sorting would interleave ``layers.10.weight`` between
    ``layers.1.*`` and ``layers.2.*`` and silently fragment the blocks.
    """
    return tuple(name for name, _ in module.named_parameters())


def flatten(params: ParamDict, names: tuple[str, ...]) -> torch.Tensor:
    """Concatenate the parameters into one vector, in ``names`` order."""
    missing = [name for name in names if name not in params]
    if missing:
        raise FunctionalError(f"parameters missing: {missing}")
    return torch.cat([params[name].reshape(-1) for name in names])


def unflatten(
    vector: torch.Tensor,
    names: tuple[str, ...],
    shapes: tuple[torch.Size, ...],
) -> ParamDict:
    """Split a flat vector back into named tensors."""
    if vector.ndim != 1:
        raise FunctionalError(f"expected a flat vector, got shape {tuple(vector.shape)}")
    expected = sum(int(torch.tensor(shape).prod()) if shape else 1 for shape in shapes)
    if vector.numel() != expected:
        raise FunctionalError(
            f"vector has {vector.numel()} entries but the model has {expected} parameters"
        )

    params: ParamDict = {}
    offset = 0
    for name, shape in zip(names, shapes, strict=True):
        size = int(torch.Size(shape).numel())
        params[name] = vector[offset : offset + size].view(shape)
        offset += size
    return params


def call(module: nn.Module, params: ParamDict, x: torch.Tensor) -> torch.Tensor:
    """Evaluate ``module`` at ``params``, leaving the module untouched."""
    return functional_call(module, params, (x,))


def vector_jacobian_product(
    module: nn.Module, params: ParamDict, x: torch.Tensor, cotangent: torch.Tensor
) -> ParamDict:
    """$\\bm u^{\\mathsf T}\\bm H$ -- one reverse-mode sweep.

    ``cotangent`` has the shape of the output. For a scalar-output model this
    single call gives the whole Jacobian row; for $q$ outputs it takes $q$ of
    them, which is the cheaper direction whenever $q < p$ (always, here).
    """

    def evaluate(p: ParamDict) -> torch.Tensor:
        return call(module, p, x)

    output, pullback = vjp(evaluate, params)
    if cotangent.shape != output.shape:
        raise FunctionalError(
            f"cotangent shape {tuple(cotangent.shape)} does not match the output "
            f"{tuple(output.shape)}"
        )
    return pullback(cotangent)[0]


def jacobian_vector_product(
    module: nn.Module, params: ParamDict, x: torch.Tensor, tangent: ParamDict
) -> torch.Tensor:
    """$\\bm H\\bm v$ -- one forward-mode sweep.

    ``tangent`` is a direction in *parameter* space, so it has the same keys and
    shapes as ``params``. Cheaper than ``vjp`` when $p < q$, which does not
    happen here but does for the dense-output CNN case (research note §7.2).
    """

    def evaluate(p: ParamDict) -> torch.Tensor:
        return call(module, p, x)

    _, tangent_out = jvp(evaluate, (params,), (tangent,))
    return tangent_out


def jacobian(
    module: nn.Module, params: ParamDict, x: torch.Tensor, names: tuple[str, ...]
) -> torch.Tensor:
    """The full $\\bm H$, materialised, shape ``(*output_shape, p)``.

    **For tests and small models only.** Built from $q$ reverse-mode sweeps. At
    $p\\sim10^5$ this is a $10^6$-entry matrix per sample and the filter must use
    the products above instead.
    """
    output = call(module, params, x)
    rows = []
    flat_output = output.reshape(-1)
    for index in range(flat_output.numel()):
        seed = torch.zeros_like(flat_output)
        seed[index] = 1.0
        grads = vector_jacobian_product(module, params, x, seed.view(output.shape))
        rows.append(flatten(grads, names))
    return torch.stack(rows).view(*output.shape, -1)


def build_param_groups(module: nn.Module, names: tuple[str, ...]) -> tuple[ParamGroup, ...]:
    """Group the flat vector by layer.

    A layer's weight and bias share a block, because they share a curvature
    block in every structured covariance the research note considers.
    """
    shapes = dict(module.named_parameters())
    spans: dict[str, list[int]] = {}
    offset = 0
    for name in names:
        size = shapes[name].numel()
        layer = name.rsplit(".", 1)[0] if "." in name else name
        span = spans.setdefault(layer, [offset, offset])
        span[1] = offset + size
        offset += size

    groups = []
    for layer, (start, stop) in spans.items():
        weight = shapes.get(f"{layer}.weight")
        fan_out, fan_in = (
            (weight.shape[0], weight.shape[1])
            if weight is not None and weight.ndim >= 2
            else (0, 0)
        )
        groups.append(
            ParamGroup(name=layer, start=start, stop=stop, fan_in=fan_in, fan_out=fan_out)
        )
    return tuple(groups)
