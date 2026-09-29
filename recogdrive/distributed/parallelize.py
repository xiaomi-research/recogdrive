"""Wrap a model for training: activation checkpointing, then per-block compile, then FSDP2 or DDP.

Frozen parameters under a `replicate_frozen` module path are left out of FSDP2 entirely
(ignored_params, or detached while wrapping on torch<2.7): no all-gather, no dtype cast,
no sharding. Every other parameter is managed
by FSDP2, including frozen ones on the trainable compute path, so they are cast to the compute
dtype together with their neighbours.
"""

import inspect
import logging
from collections.abc import Mapping
from typing import Dict, Iterable, List, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def fp32_average_hook(process_group, bucket):
    """DDP comm hook: bf16 gradients are averaged in fp32, not summed in bf16."""
    buffer = bucket.buffer()
    wide = buffer.float().div_(dist.get_world_size(process_group))
    future = dist.all_reduce(wide, group=process_group, async_op=True).get_future()
    return future.then(lambda done: buffer.copy_(done.value()[0]))


def expose_output(module: nn.Module, args, output) -> None:
    """Forward hook: FSDP2 (and DDP) hook the backward pass onto the tensors they find in the forward output through
    torch pytree. A container pytree does not know (transformers' BatchFeature, SimpleNamespace) hides the loss, and
    FSDP2 then leaves the parameters unsharded without ever reducing their gradients. Such a container type is
    registered with pytree the first time a model returns it."""
    kind = type(output)
    if kind in pytree.SUPPORTED_NODES or torch.is_tensor(output) or output is None:
        return
    if isinstance(output, Mapping):
        pytree.register_pytree_node(kind, lambda m: (list(m.values()), list(m.keys())),
                                    lambda values, keys: kind(dict(zip(keys, values))))
    elif hasattr(output, "__dict__"):
        pytree.register_pytree_node(kind, lambda o: (list(vars(o).values()), list(vars(o))),
                                    lambda values, keys: kind(**dict(zip(keys, values))))


def base_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


def under(name: str, patterns: Iterable[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in patterns)


def replicated_params(model: nn.Module, replicate_frozen: Iterable[str]) -> Set[nn.Parameter]:
    """Frozen parameters kept whole on every rank, plus 0-dim parameters FSDP2 cannot shard."""
    patterns = list(replicate_frozen)
    out: Set[nn.Parameter] = set()
    trainable_scalars = []
    for name, param in model.named_parameters():
        if param.ndim == 0:
            if param.requires_grad:
                trainable_scalars.append(name)
            out.add(param)
        elif not param.requires_grad and under(name, patterns):
            out.add(param)
    if trainable_scalars:
        raise ValueError(
            "0-dim trainable parameters cannot be sharded and would train differently on every rank: "
            + ", ".join(trainable_scalars[:8])
            + ". Give them shape (1,)."
        )
    return out


def block_runs(model: nn.Module, wrap_classes: Iterable[str], skip: Set[nn.Parameter]) -> List[List[nn.Module]]:
    """Repeated blocks in execution order, one run per container.

    With wrap_classes empty, a block is an element of a ModuleList/Sequential whose children all
    share one class, or a module whose class is listed in a HuggingFace `_no_split_modules`.
    Blocks whose parameters are all in `skip` are dropped: there is nothing to shard.
    """
    wanted = set(wrap_classes)
    no_split = set()
    if not wanted:
        for module in model.modules():
            no_split.update(getattr(module, "_no_split_modules", None) or [])

    def managed(module: nn.Module) -> bool:
        return any(p not in skip for p in module.parameters())

    runs: List[List[nn.Module]] = []
    seen = set()
    for module in model.modules():
        if not isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue
        children = list(module.children())
        if wanted:
            blocks = [c for c in children if type(base_module(c)).__name__ in wanted]
        elif len(children) >= 2 and len({type(base_module(c)) for c in children}) == 1:
            blocks = children
        else:
            blocks = [c for c in children if type(base_module(c)).__name__ in no_split]
        blocks = [b for b in blocks if id(b) not in seen and managed(b)]
        if blocks:
            seen.update(id(b) for b in blocks)
            runs.append(blocks)
    for module in model.modules():
        name = type(base_module(module)).__name__
        if id(module) not in seen and (name in wanted or name in no_split) and managed(module):
            seen.add(id(module))
            runs.append([module])
    return runs


def apply_activation_checkpointing(model: nn.Module, class_names: Iterable[str], limit: Optional[int] = None) -> int:
    """Recomputes the trainable modules of these classes in backward: all of them, or the first `limit` in module
    order. Every block left out keeps its activations, trading memory for one less forward in backward."""
    names = set(class_names)
    if not names:
        return 0
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

    targets = [
        (qualname, module)
        for qualname, module in model.named_modules()
        if qualname and type(module).__name__ in names and any(p.requires_grad for p in module.parameters())
    ]
    selected = []
    for qualname, module in targets:
        if not any(qualname.startswith(parent + ".") for parent, _ in selected):
            selected.append((qualname, module))
    for qualname, module in selected[:limit]:
        parent_name, _, child = qualname.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT))
    return len(selected[:limit])


def compile_blocks(runs: List[List[nn.Module]]) -> None:
    # Compiling the innermost forward in place keeps FSDP hooks and checkpoint wrappers outside the graph and
    # state_dict keys unchanged. A graph traced through a checkpoint wrapper would also freeze the wrapped layer's
    # forward, and transformers>=4.56 swaps that forward on every call to record output_hidden_states.
    for run in runs:
        for block in run:
            inner = base_module(block)
            inner.forward = torch.compile(inner.forward)


def broadcast_params(params: List[torch.Tensor]) -> None:
    works = [dist.broadcast(p.data, src=0, async_op=True) for p in params]
    for work in works:
        work.wait()


def module_depth(model: nn.Module) -> Dict[int, int]:
    return {id(m): (name.count(".") + 1 if name else 0) for name, m in model.named_modules()}


def replicated_holders(model: nn.Module, replicated: Set[nn.Parameter]):
    """Outermost submodules whose parameters are all replicated, as (parent, name, module)."""
    holders, taken = [], []
    for qualname, module in model.named_modules():
        if not qualname or any(qualname.startswith(t + ".") for t in taken):
            continue
        params = list(module.parameters())
        if params and all(p in replicated for p in params):
            parent_name, _, child = qualname.rpartition(".")
            holders.append((model.get_submodule(parent_name) if parent_name else model, child, module))
            taken.append(qualname)
    return holders


def shard(module: nn.Module, replicated: Set[nn.Parameter], **kwargs) -> None:
    from torch.distributed.fsdp import fully_shard

    if "ignored_params" in inspect.signature(fully_shard).parameters:
        fully_shard(module, ignored_params=replicated, **kwargs)
        return
    # torch<2.7: keep replicated modules out of the FSDP group by detaching them while it is built.
    holders = replicated_holders(module, replicated)
    inside = {p for p in module.parameters() if p in replicated}
    if inside - {p for _, _, m in holders for p in m.parameters()}:
        raise RuntimeError("torch<2.7 can only keep whole frozen modules out of FSDP2; install torch>=2.7")
    parents = {id(parent): (parent, list(parent._modules.items())) for parent, _, _ in holders}
    for parent, name, _ in holders:
        del parent._modules[name]
    try:
        fully_shard(module, **kwargs)
    finally:
        # restore the registration order: parameters() order decides e.g. which dtype a model reads from its first weight
        for parent, children in parents.values():
            parent._modules.clear()
            parent._modules.update(children)


def set_prefetch(runs: List[List[nn.Module]], distance: int) -> int:
    if distance <= 0:
        return 0
    from torch.distributed.fsdp import FSDPModule

    edges = 0
    for run in runs:
        blocks = [b for b in run if isinstance(b, FSDPModule)]
        for i, block in enumerate(blocks):
            forward = blocks[i + 1 : i + 1 + distance]
            backward = list(reversed(blocks[max(0, i - distance) : i]))
            if forward:
                block.set_modules_to_forward_prefetch(forward)
            if backward:
                block.set_modules_to_backward_prefetch(backward)
            edges += len(forward) + len(backward)
    return edges


def parallelize(model: nn.Module, args, ctx) -> nn.Module:
    """Returns the module to call for forward. For FSDP2 it is `model` itself, for DDP a wrapper."""
    model.to(ctx.device)
    model.register_forward_hook(expose_output)  # before FSDP2 / DDP register theirs, so it runs first
    replicated = replicated_params(model, args.replicate_frozen)
    # FSDP2 keeps fp32 master shards and computes in args.precision; under DDP the parameters are the compute
    # dtype, and an optimizer with fp32 master copies (Muon) keeps bf16 training exact. Frozen parameters outside
    # replicate_frozen compute in args.precision either way (FSDP2 casts them), so modules mixing trainable and
    # frozen ones (a frozen reference policy fed the policy's features) see one dtype.
    master = torch.float32 if args.strategy == "fsdp" else DTYPES[args.precision]
    with torch.no_grad():
        for param in model.parameters():
            cast = param.requires_grad if args.strategy == "fsdp" else param not in replicated
            if cast and param.is_floating_point() and param.dtype != master:
                param.data = param.data.to(master)
    managed = [p for p in model.parameters() if p not in replicated]
    if not any(p.requires_grad for p in managed):
        raise ValueError("no trainable parameters to optimize")
    if ctx.world_size > 1:
        broadcast_params(managed)

    wrapped_ac = apply_activation_checkpointing(model, args.activation_checkpointing, args.activation_checkpointing_layers)
    runs = block_runs(model, args.fsdp_wrap_modules, replicated)
    compiled = block_runs(model, args.fsdp_wrap_modules, set()) if args.compile else []
    compile_blocks(compiled)
    nblocks = sum(len(r) for r in runs)
    ncompiled = sum(len(r) for r in compiled)
    replicated_gib = sum(p.numel() * p.element_size() for p in replicated) / 1024 ** 3

    if args.strategy == "ddp":
        if ctx.world_size == 1:
            wrapper = model
        else:
            kwargs = dict(
                device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                output_device=ctx.local_rank if ctx.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=args.ddp_find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=args.ddp_static_graph,
                bucket_cap_mb=args.ddp_bucket_cap_mb,
            )
            if "init_sync" in inspect.signature(DDP.__init__).parameters:
                kwargs["init_sync"] = False
            wrapper = DDP(model, **kwargs)
            if master != torch.float32:
                wrapper.register_comm_hook(None, fp32_average_hook)
        if ctx.is_main:
            logger.info(
                "parallel=ddp world=%d precision=%s blocks=%d compiled_blocks=%d activation_checkpointing=%d "
                "replicated_frozen=%.2fGiB", ctx.world_size, args.precision, nblocks, ncompiled, wrapped_ac, replicated_gib,
            )
        return wrapper

    from torch.distributed.fsdp import MixedPrecisionPolicy

    mesh = ctx.build_mesh(args.hsdp_shard_size)
    compute = DTYPES[args.precision]
    param_dtype = None if compute == torch.float32 else compute
    block_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    # Model inputs keep their dtype; the model casts them to its parameter dtype itself.
    root_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, cast_forward_inputs=False)
    depth = module_depth(model)
    for block in sorted((b for run in runs for b in run), key=lambda m: depth.get(id(m), 0), reverse=True):
        shard(block, replicated, mesh=mesh, reshard_after_forward=args.reshard_after_forward, mp_policy=block_policy)
    shard(model, replicated, mesh=mesh, reshard_after_forward=args.reshard_after_forward, mp_policy=root_policy)
    edges = set_prefetch(runs, args.prefetch_distance)
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    if ctx.is_main:
        logger.info(
            "parallel=fsdp2 mesh=%s precision=%s reshard_after_forward=%s blocks=%d prefetch_edges=%d "
            "compiled_blocks=%d activation_checkpointing=%d replicated_frozen=%.2fGiB",
            tuple(mesh.shape), args.precision, args.reshard_after_forward, nblocks, edges,
            ncompiled, wrapped_ac, replicated_gib,
        )
    return model


def gradient_sync(wrapper: nn.Module, sync: bool):
    """Context that skips the gradient reduction on accumulation micro-steps."""
    import contextlib

    if sync:
        return contextlib.nullcontext()
    if isinstance(wrapper, DDP):
        return wrapper.no_sync()
    setter = getattr(wrapper, "set_requires_gradient_sync", None)
    if setter is None:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def fsdp_no_sync():
        setter(False)
        try:
            yield
        finally:
            setter(True)

    return fsdp_no_sync()


def unwrap(wrapper: nn.Module) -> nn.Module:
    return wrapper.module if isinstance(wrapper, DDP) else wrapper
