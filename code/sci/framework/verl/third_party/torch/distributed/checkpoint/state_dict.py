# official torch 2.6.0 set_model_state_dict API leads to OOM
# this is a copy of torch/distributed/checkpoint from torch 2.7.0

# From PyTorch:

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions by Tri Dao:
# Copyright (c) 2024 Tri Dao.
# All rights reserved.
import contextlib
import functools
import gc
import warnings
from collections.abc import Generator, Iterable
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Any, Callable, Optional, Union, cast, no_type_check

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
)
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp._common_utils import (
    FSDP_WRAPPED_MODULE,
    _get_module_fsdp_state_if_fully_sharded_module,
)
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils._pytree import tree_map_only

from verl.third_party.torch.distributed._state_dict_utils import (
    _broadcast_state_dict,
    _distribute_state_dict,
    _flatten_state_dict,
    _gather_state_dict,
    _offload_state_dict_to_cpu,
    _unflatten_state_dict,
)

__all__ = [
    "FQNS_T",
    "PrimitiveType",
    "ValueType",
    "DictValueType",
    "ListDictValueType",
    "OptimizerStateType",
    "StateDictOptions",
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "get_state_dict",
    "set_model_state_dict",
    "set_optimizer_state_dict",
    "set_state_dict",
]


_FLAT_PARAM = "_flat_param"
_PG = "param_groups"
_PARAMS = "params"
_STATE = "state"

FQNS_T = set[str]
PrimitiveType = Union[DTensor, ShardedTensor, torch.Tensor, int, float, str]
ValueType = Union[PrimitiveType, list[PrimitiveType], tuple[PrimitiveType], dict[str, "ValueType"]]
DictValueType = dict[str, ValueType]
ListDictValueType = list[DictValueType]
OptimizerStateType = dict[str, DictValueType | ListDictValueType]


_patched_state_dict: set[Callable] = set()


@contextlib.contextmanager
def _gc_context():
    is_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if is_enabled:
            gc.enable()


@dataclass
class StateDictOptions:

    full_state_dict: bool = False
    cpu_offload: bool = False
    ignore_frozen_params: bool = False
    keep_submodule_prefixes: bool = True
    strict: bool = True
    broadcast_from_rank0: bool = False
    flatten_optimizer_state_dict: bool = False
    dsd_fqn_modifiers: str = "_fqn_modifiers"


@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: dict[
        str | torch.Tensor,
        FQNS_T | torch.Tensor,
    ] = field(default_factory=dict)
    shared_params_mapping: dict[
        str | torch.Tensor,
        FQNS_T | torch.Tensor,
    ] = field(default_factory=dict)
    submodule_prefixes: set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable = contextlib.nullcontext
    fsdp_modules: list[nn.Module] = field(default_factory=list)


@functools.cache
def _get_fqns(
    model: nn.Module,
    name: str,
    dsd_fqn_modifiers: str = "_fqn_modifiers",
    skip_ddp_prefix: bool = True,
    skip_compiler_prefix: bool = True,
) -> FQNS_T:

    name = name.replace(_CHECKPOINT_PREFIX, "")
    if "." not in name:
        return {name}

    obj_names = name.split(".")
    fqn_obj_names = []
    curr_obj = model
    for i, curr_obj_name in enumerate(obj_names):
        if isinstance(curr_obj, DDP):
            assert curr_obj_name == "module"
            curr_obj = curr_obj.module
            if not skip_ddp_prefix:
                fqn_obj_names.append(curr_obj_name)
        elif isinstance(curr_obj, FSDP):
            if i < len(obj_names) - 1 and obj_names[i + 1] == _FLAT_PARAM:
                prefix = ".".join(fqn_obj_names)
                flat_param = getattr(curr_obj, _FLAT_PARAM)
                if prefix:
                    prefix = f"{prefix}."
                return {f"{prefix}{fqn}" for fqn in flat_param._fqns}
            curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
            if curr_obj_name != FSDP_WRAPPED_MODULE:
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)
        elif isinstance(curr_obj, torch._dynamo.eval_frame.OptimizedModule):
            assert curr_obj_name == "_orig_mod"
            curr_obj = curr_obj._orig_mod
            if not skip_compiler_prefix:
                fqn_obj_names.append(curr_obj_name)
        else:
            if hasattr(curr_obj, dsd_fqn_modifiers):
                if removed_fqn := getattr(curr_obj, dsd_fqn_modifiers)().get(curr_obj_name):
                    if hasattr(curr_obj, removed_fqn):
                        curr_obj = getattr(curr_obj, removed_fqn)
            fqn_obj_names.append(curr_obj_name)
            if curr_obj_name == nn.modules.module._EXTRA_STATE_KEY_SUFFIX:
                if i != len(obj_names) - 1:
                    raise RuntimeError("Expect `_extra_state` to be the last obj name")
            else:
                curr_obj = getattr(curr_obj, curr_obj_name)

    return {".".join(fqn_obj_names).replace(_CHECKPOINT_PREFIX, "")}


class _EXTRA_STATE:
    pass


def _iterate_valid_model_state(model, dsd_fqn_modifiers="_fqn_modifiers"):
    visited_modules: set[nn.Module] = set()

    def recurse(module: nn.Module, curr_fqn: str) -> Generator:
        visited_modules.add(module)

        curr_fqn = f"{curr_fqn}." if curr_fqn else ""
        for name, submodule in module.named_children():
            if submodule in visited_modules:
                continue
            if hasattr(module, dsd_fqn_modifiers) and name in getattr(module, dsd_fqn_modifiers)().values():
                new_fqn = curr_fqn[:-1]
            else:
                new_fqn = f"{curr_fqn}{name}"
            yield from recurse(submodule, new_fqn)

        for name, obj in chain(module.named_buffers(recurse=False), module.named_parameters(recurse=False)):
            if name in module._non_persistent_buffers_set:
                continue
            new_fqn = f"{curr_fqn}{name}"
            yield new_fqn, obj

        if getattr(module.__class__, "get_extra_state", nn.Module.get_extra_state) != nn.Module.get_extra_state:
            new_fqn = f"{curr_fqn}{nn.modules.module._EXTRA_STATE_KEY_SUFFIX}"
            yield new_fqn, _EXTRA_STATE()

    yield from recurse(model, "")


def _verify_options(
    model: nn.Module,
    optims: tuple[torch.optim.Optimizer, ...],
    optim_only: bool,
    *,
    submodules: Optional[set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> _StateDictInfo:
    if submodules:
        warnings.warn(
            "Getting submodules only model/optim state_dict is deprecated and "
            "will be removed in 2.5. This feature can be achieved by manually "
            "filtering out the state_dict returned from get_state_dict.",
            FutureWarning,
        )
    if optim_only and not optims:
        raise RuntimeError("Optimizers are not passed in but optim_only is set to True.")

    options = options or StateDictOptions()

    fqn_param_mapping: dict[str | torch.Tensor, set[str] | torch.Tensor] = {}
    shared_params_mapping: dict[str | torch.Tensor, set[str] | torch.Tensor] = {}
    for name, param in _iterate_valid_model_state(model):
        if isinstance(param, _EXTRA_STATE):
            continue

        fqns = _get_fqns(model, name)
        fqn = fqn_param_mapping.get(param, None)
        if fqn is not None:
            cast(set[str], fqn_param_mapping[param]).update(fqns)
            shared_params_mapping[param] = fqn_param_mapping[param]
        else:
            fqn_param_mapping[param] = fqns.copy()
        for fqn in fqns:
            if not isinstance(param, _EXTRA_STATE):
                fqn_param_mapping[fqn] = param

    for param_, fqns_ in list(shared_params_mapping.items()):
        for fqn in fqns_:
            shared_params_mapping[fqn] = cast(torch.Tensor, param_)

    submodule_prefixes: set[str] = set()
    if submodules:
        submodules = set(submodules)
        for name, module in model.named_modules():
            if module not in submodules:
                continue
            fqns = _get_fqns(model, name)
            assert len(fqns) == 1, "Submodule FQN should only have 1 instance"
            submodule_prefixes.update(f"{fqn}." for fqn in fqns)

    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError("full_state_dict must be True when broadcast_from_rank0 is True.")
    fsdp_modules = FSDP.fsdp_modules(model)
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
    fsdp_context: Callable
    if fsdp_modules:
        if options.full_state_dict:
            state_dict_config = FullStateDictConfig(offload_to_cpu=options.cpu_offload, rank0_only=options.cpu_offload)
            optim_state_dict_config = FullOptimStateDictConfig(
                offload_to_cpu=options.cpu_offload,
                rank0_only=(options.cpu_offload or options.broadcast_from_rank0),
            )
            state_dict_type = StateDictType.FULL_STATE_DICT
        else:
            state_dict_config = ShardedStateDictConfig(
                offload_to_cpu=options.cpu_offload,
            )
            optim_state_dict_config = ShardedOptimStateDictConfig(
                offload_to_cpu=options.cpu_offload,
            )
            state_dict_type = StateDictType.SHARDED_STATE_DICT

        @contextlib.contextmanager
        def fsdp_state_dict_type_without_warning(
            module,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FSDP.state_dict_type", category=FutureWarning)
                with FSDP.state_dict_type(
                    module=module,
                    state_dict_type=state_dict_type,
                    state_dict_config=state_dict_config,
                    optim_state_dict_config=optim_state_dict_config,
                ):
                    yield

        fsdp_context = functools.partial(
            fsdp_state_dict_type_without_warning,
            module=model,
            state_dict_type=state_dict_type,
            state_dict_config=state_dict_config,
            optim_state_dict_config=optim_state_dict_config,
        )
    else:
        fsdp_context = contextlib.nullcontext

    return _StateDictInfo(
        **asdict(options),
        fqn_param_mapping=fqn_param_mapping,
        shared_params_mapping=shared_params_mapping,
        submodule_prefixes=submodule_prefixes,
        fsdp_context=fsdp_context,
        fsdp_modules=cast(list[nn.Module], fsdp_modules),
        handle_model=not optim_only,
        handle_optim=(len(optims) > 0),
    )


def _verify_state_dict(
    model_state_dict: dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    for module in info.fsdp_modules:
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        assert fsdp_state is not None, "Expected a fsdp_state with a fsdp module."

    if (
        info.handle_model
        and not model_state_dict
        and not info.submodule_prefixes
        and not info.ignore_frozen_params
        and not (info.cpu_offload and info.full_state_dict)
        and info.strict
        and not info.broadcast_from_rank0
    ):
        raise RuntimeError(
            "The option indicates that model state_dict is required to save "
            "or load, but model state_dict is empty."
            f"rank = {dist.get_rank()=}."
        )

    if info.handle_optim:
        if not optim_state_dict and not (info.cpu_offload and info.full_state_dict) and (not info.broadcast_from_rank0):
            raise RuntimeError(
                "The option indicates that model state_dict is required to save, "
                f"or load but optim state_dict is empty. {optim_state_dict}"
            )

    for key in model_state_dict.keys():
        if _FLAT_PARAM in key:
            raise RuntimeError(f"{key} contains {_FLAT_PARAM}. This can happen if the model is not the root module.")


def _state_dict_fn(obj: nn.Module | torch.optim.Optimizer, api: str) -> Callable:
    call = getattr(obj, api)
    if call in _patched_state_dict:
        call = functools.partial(getattr(obj.__class__, api), self=obj)
    return call


def _maybe_full_or_cpu_state_dict(state_dict: dict[str, Any], info: _StateDictInfo) -> dict[str, Any]:
    if info.full_state_dict:
        ranks_only = () if (not info.cpu_offload or not torch.distributed.is_initialized()) else (0,)
        return _gather_state_dict(state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only)
    elif info.cpu_offload:
        return _offload_state_dict_to_cpu(state_dict)
    else:
        return state_dict


@torch.no_grad()
def _get_model_state_dict(model: nn.Module, info: _StateDictInfo) -> dict[str, ValueType]:
    if not info.handle_model:
        return {}

    with info.fsdp_context():
        state_dict = _state_dict_fn(model, "state_dict")()

    for key in list(state_dict.keys()):
        fqns = _get_fqns(model, key)
        assert len(fqns) == 1, (key, fqns)
        fqn = next(iter(fqns))
        if fqn != key:
            def verify(key, fqn) -> bool:
                if len(fqn) >= len(key):
                    return False
                fqn_split = fqn.split(".")
                key_split = key.split(".")
                fqn_idx = 0
                for key_idx, key_name in enumerate(key_split):
                    if key_name == fqn_split[fqn_idx]:
                        fqn_idx += 1
                        if fqn_idx == len(fqn_split):
                            return key_idx == len(key_split) - 1
                    elif key_name in ("module", "_orig_mod"):
                        continue
                    else:
                        return False
                return True

            if not verify(key, fqn):
                raise RuntimeError(f"An unexpected key, {key}, exists. FQN is {fqn}")
            state_dict[fqn] = state_dict.pop(key)

    if info.submodule_prefixes:
        new_state_dict: dict[str, ValueType] = {}
        for fqn in state_dict.keys():
            for prefix in info.submodule_prefixes:
                if not fqn.startswith(prefix):
                    continue
                if info.keep_submodule_prefixes:
                    new_state_dict[fqn] = state_dict[fqn]
                else:
                    new_fqn = fqn[len(prefix) :]
                    new_state_dict[new_fqn] = state_dict[fqn]
        state_dict = new_state_dict

    if info.ignore_frozen_params:
        for key, param in model.named_parameters():
            if param.requires_grad:
                continue
            fqns = _get_fqns(model, key)
            for fqn in fqns:
                state_dict.pop(fqn)

    for key, p in list(state_dict.items()):
        if torch.is_tensor(p) and p.is_meta:
            state_dict.pop(key)

    return _maybe_full_or_cpu_state_dict(state_dict, info)


@torch.no_grad()
def _load_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, ValueType],
    info: _StateDictInfo,
) -> _IncompatibleKeys:
    if not info.handle_model or (not state_dict and not info.broadcast_from_rank0):
        return _IncompatibleKeys({}, {})

    local_state_dict = {}
    for key, value in _iterate_valid_model_state(model, info.dsd_fqn_modifiers):
        fqns = _get_fqns(model, key, info.dsd_fqn_modifiers)
        fqns_with_prefix = _get_fqns(
            model,
            key,
            info.dsd_fqn_modifiers,
            skip_ddp_prefix=False,
            skip_compiler_prefix=False,
        )

        for fqn, fqn_with_prefix in zip(fqns, fqns_with_prefix, strict=False):
            if (not info.broadcast_from_rank0 or dist.get_rank() == 0) and fqn != fqn_with_prefix:
                load_value = state_dict.pop(fqn, None)
                if load_value is None:
                    if info.strict:
                        raise RuntimeError(f"Missing key: {fqn}.")
                else:
                    state_dict[fqn_with_prefix] = load_value
            local_state_dict[fqn_with_prefix] = value

    assign = False
    if info.broadcast_from_rank0 or info.full_state_dict:
        devices = set()
        for key, value in local_state_dict.items():
            if torch.is_tensor(value) and value.dim() > 0:
                devices.add(value.device)
        if torch.device("meta") in devices:
            devices.remove(torch.device("meta"))
            assign = True
        if len(devices) == 0:
            devices.add(dist.distributed_c10d._get_pg_default_device())
        elif len(devices) > 1:
            raise ValueError("Multiple devices found")

        if info.broadcast_from_rank0:
            _broadcast_state_dict(
                state_dict,
                local_state_dict,
                device=devices.pop(),
                strict=info.strict,
                cpu_offload=info.cpu_offload,
            )
        elif info.full_state_dict:
            _distribute_state_dict(state_dict, local_state_dict, device=devices.pop())
        for fqn, local_state in local_state_dict.items():
            state_dict[fqn] = local_state

    with info.fsdp_context():
        return cast(
            _IncompatibleKeys,
            _state_dict_fn(model, "load_state_dict")(state_dict=state_dict, strict=info.strict, assign=assign),
        )


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    if optim.state:
        return

    for param_group in optim.param_groups:
        for param in param_group[_PARAMS]:
            if param.grad is not None:
                return

    for param_group in optim.param_groups:
        for param in param_group[_PARAMS]:
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

    lrs = []
    for param_group in optim.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
            param_group["lr"] = torch.tensor(0.0) if isinstance(param_group["lr"], torch.Tensor) else 0.0
    optim.step(closure=None)
    for param_group in optim.param_groups:
        if "lr" in param_group:
            param_group["lr"] = lrs.pop(0)
    optim.zero_grad(set_to_none=True)


def _flatten_optim_state_dict(state_dict: OptimizerStateType) -> dict[str, ValueType]:

    def _raise_if_type_not_supported(v):
        if not isinstance(v, (torch.Tensor, int, float)):
            raise NotImplementedError(
                f"Flattening optimizer state_dict only supports tensor, int, float states now. Type is {type(v)}."
            )

    ret: dict[str, ValueType] = {}
    for fqn, state in cast(DictValueType, state_dict[_STATE]).items():
        for k, v in cast(DictValueType, state).items():
            _raise_if_type_not_supported(v)
            ret[f"{_STATE}.{fqn}.{k}"] = v

    for param_group in cast(ListDictValueType, state_dict[_PG]):
        fqns = param_group.pop(_PARAMS)
        for fqn in cast(list[str], fqns):
            for k, v in param_group.items():
                ret[f"{_PG}.{fqn}.{k}"] = v
    return ret


def _unflatten_optim_state_dict(
    optim: torch.optim.Optimizer,
    state_dict: dict[str, ValueType],
    info: _StateDictInfo,
) -> OptimizerStateType:
    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}

    for param_group in optim.param_groups:
        pg_state.append({_PARAMS: []})
        for param in param_group[_PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                if fqn in info.shared_params_mapping:
                    in_params = False
                    for k in param_group.keys():
                        if k == _PARAMS:
                            continue
                        flatten_key = f"{_PG}.{fqn}.{k}"
                        if flatten_key in state_dict:
                            in_params = True
                        break
                else:
                    in_params = True

                if not in_params:
                    continue

                params = pg_state[-1][_PARAMS]
                assert isinstance(params, list)
                params.append(fqn)
                if not param.requires_grad:
                    continue
                state[fqn] = {}
                for state_name in optim.state[param].keys():
                    cast(DictValueType, state[fqn])[state_name] = state_dict[f"{_STATE}.{fqn}.{state_name}"]

        first_param_fqn = cast(list[str], pg_state[-1][_PARAMS])[0]
        for k in param_group.keys():
            if k == _PARAMS:
                continue
            value = state_dict[f"{_PG}.{first_param_fqn}.{k}"]
            if k not in pg_state[-1]:
                pg_state[-1][k] = value
            elif pg_state[-1][k] != value:
                raise RuntimeError(
                    "All the parameters in the same parameter group should have "
                    f"the same saved param_group value. But {first_param_fqn}.{k} "
                    f"is {value} while other(s) is {pg_state[-1][k]}."
                )

    return return_osd


@torch.no_grad()
def _get_optim_state_dict(
    model: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, ...],
    info: _StateDictInfo,
) -> OptimizerStateType:
    if not info.handle_optim:
        return {}

    optim_state_dict: OptimizerStateType = {_STATE: {}, _PG: []}
    for optim in optimizers:
        _init_optim_state(optim)
        osd = _state_dict_fn(optim, "state_dict")()
        if info.fsdp_modules:
            with info.fsdp_context():
                osd = FSDP.optim_state_dict(model, optim, osd)

            if not osd:
                continue
            for k in list(osd[_STATE].keys()):
                if "_orig_mod" in k:
                    osd[_STATE][k.replace("_orig_mod.", "")] = osd[_STATE].pop(k)
            for g in osd[_PG]:
                params = [k.replace("_orig_mod.", "") for k in g[_PARAMS]]
                g[_PARAMS] = params
        else:
            params = list(chain.from_iterable(g[_PARAMS] for g in optim.param_groups))
            param_pid_mapping = dict(zip(params, range(len(params)), strict=False))
            fqn_pid_mapping = {}
            for key, param in model.named_parameters():
                fqns = _get_fqns(model, key)
                assert len(fqns) == 1
                fqn = next(iter(fqns))
                if param not in param_pid_mapping:
                    continue
                pid = param_pid_mapping[param]
                fqn_pid_mapping[fqn] = pid
                fqn_pid_mapping[pid] = fqn

            for key in list(osd[_STATE].keys()):
                fqn = fqn_pid_mapping[key]
                osd[_STATE][fqn] = osd[_STATE].pop(key)

            for group in osd[_PG]:
                group[_PARAMS] = [fqn_pid_mapping[pid] for pid in group[_PARAMS]]

        if not osd:
            continue

        cast(DictValueType, optim_state_dict[_STATE]).update(osd[_STATE])
        cast(ListDictValueType, optim_state_dict[_PG]).extend(osd[_PG])

    if info.flatten_optimizer_state_dict:
        optim_state_dict = cast(OptimizerStateType, _flatten_optim_state_dict(optim_state_dict))

    return _maybe_full_or_cpu_state_dict(optim_state_dict, info)


def _split_optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> OptimizerStateType:

    state: DictValueType = {}
    pg_state: ListDictValueType = []
    return_osd: OptimizerStateType = {_STATE: state, _PG: pg_state}
    pg_mapping: dict[int, int] = {}

    if all(isinstance(k, int) for k in cast(DictValueType, optim_state_dict[_STATE]).keys()):
        return optim_state_dict

    for param_group in optim.param_groups:
        pg_state.append({_PARAMS: []})
        for param in param_group[_PARAMS]:
            for fqn in info.fqn_param_mapping[param]:
                if fqn in info.shared_params_mapping:
                    in_params = False
                    for loaded_param_group in cast(ListDictValueType, optim_state_dict[_PG]):
                        if fqn in cast(list[str], loaded_param_group[_PARAMS]):
                            in_params = True
                            break
                else:
                    in_params = True
                if not in_params:
                    continue

                params = pg_state[-1][_PARAMS]
                assert isinstance(params, list)
                params.append(fqn)
                if param.requires_grad:
                    state[fqn] = cast(DictValueType, optim_state_dict[_STATE])[fqn]
                for loaded_param_group in cast(ListDictValueType, optim_state_dict[_PG]):
                    if fqn in cast(list[str], loaded_param_group[_PARAMS]):
                        pg_mapping[id(loaded_param_group)] = len(return_osd[_PG]) - 1

        if len(param_group[_PARAMS]) == 0:
            ret = []
            for loaded_param_group in cast(ListDictValueType, optim_state_dict[_PG]):
                if len(cast(list[str], loaded_param_group[_PARAMS])) == 0:
                    ret.append(loaded_param_group)
            if len(ret) != 1:
                raise ValueError(
                    "There are param groups that have zero parameters. "
                    "In such a case, DSD only support exactly one param group "
                    "with zero parameters."
                    "But the loaded state_dict has zero or more than one param groups "
                    "that have zero parameters."
                )
            if len(optim_state_dict[_PG]) != len(optim.param_groups):
                raise ValueError(
                    "When there is a parameter group that has zero parameters, multiple optimizers are not supported."
                )
            pg_mapping[id(loaded_param_group)] = len(return_osd[_PG]) - 1

    for param_group in cast(ListDictValueType, optim_state_dict[_PG]):
        pg_idx = pg_mapping.get(id(param_group), -1)
        if pg_idx == -1:
            continue

        for key, value in param_group.items():
            if key == _PARAMS:
                continue
            pg_state[pg_idx][key] = value

    return return_osd


@torch.no_grad()
def _load_optim_state_dict(
    model: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, ...],
    state_dict: OptimizerStateType,
    info: _StateDictInfo,
) -> None:
    if not info.handle_optim:
        return

    for optim in optimizers:
        _init_optim_state(optim)
        if state_dict:
            if _STATE in state_dict:
                optim_state_dict = _split_optim_state_dict(model, optim, state_dict, info)
            else:
                optim_state_dict = _unflatten_optim_state_dict(optim, cast(dict[str, ValueType], state_dict), info)
        else:
            optim_state_dict = {}
        if info.fsdp_modules:
            for original_fqn, _ in model.named_parameters():
                fqns = _get_fqns(model, original_fqn)
                fqns_with_compiler = _get_fqns(model, original_fqn, skip_compiler_prefix=False)
                if fqns == fqns_with_compiler:
                    continue

                assert len(fqns) == 1
                fqn = fqns.pop()
                fqn_with_compiler = fqns_with_compiler.pop()
                for g in optim_state_dict[_PG]:
                    val = cast(dict[str, Any], g)
                    params = [key.replace(fqn, fqn_with_compiler) for key in val[_PARAMS]]
                    val[_PARAMS] = params
                osd_state = cast(DictValueType, optim_state_dict[_STATE])
                for k in list(osd_state.keys()):
                    if fqn in k:
                        osd_state[k.replace(fqn, fqn_with_compiler)] = osd_state.pop(k)

            with info.fsdp_context():
                optim_state_dict = FSDP.optim_state_dict_to_load(model, optim, optim_state_dict)
        elif info.full_state_dict:
            info.full_state_dict = False
            local_state_dict = _get_optim_state_dict(model, (optim,), info)
            info.full_state_dict = True
            device = None

            def _device(t):
                if t.dim() > 0:
                    nonlocal device
                    if device is None:
                        device = t.device
                    elif device != t.device:
                        raise ValueError("Device mismatch")
                return t

            _ = tree_map_only(torch.Tensor, _device, local_state_dict)
            assert device is not None
            flatten_osd, osd_mapping = _flatten_state_dict(optim_state_dict)
            flatten_local_osd, local_osd_mapping = _flatten_state_dict(local_state_dict)
            if info.broadcast_from_rank0:
                _broadcast_state_dict(flatten_osd, flatten_local_osd, device=device)
            else:
                _distribute_state_dict(flatten_osd, flatten_local_osd, device=device)
            for optim_key in flatten_osd.keys():
                if optim_key not in flatten_local_osd:
                    assert optim_key in osd_mapping
                    flatten_local_osd[optim_key] = flatten_osd[optim_key]
                    local_osd_mapping[optim_key] = osd_mapping[optim_key]
            optim_state_dict = _unflatten_state_dict(flatten_local_osd, local_osd_mapping)
            for pg in optim_state_dict[_PG]:
                if _PARAMS not in pg:
                    cast(dict[str, ValueType], pg)[_PARAMS] = []

        _state_dict_fn(optim, "load_state_dict")(state_dict=optim_state_dict)


def get_model_state_dict(
    model: nn.Module,
    *,
    submodules: Optional[set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> dict[str, ValueType]:
    with _gc_context():
        info = _verify_options(
            model,
            (),
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        model_state_dict = _get_model_state_dict(model, info)
        _verify_state_dict(model_state_dict, {}, info)
        return model_state_dict


def get_optimizer_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    submodules: Optional[set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> OptimizerStateType:
    with _gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(
            model,
            optimizers,
            optim_only=True,
            submodules=submodules,
            options=options,
        )
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict({}, optim_state_dict, info)
        return optim_state_dict


def get_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    submodules: Optional[set[nn.Module]] = None,
    options: Optional[StateDictOptions] = None,
) -> tuple[dict[str, ValueType], OptimizerStateType]:

    with _gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(
            model,
            optimizers,
            optim_only=False,
            submodules=submodules,
            options=options,
        )
        model_state_dict = _get_model_state_dict(model, info)
        optim_state_dict = _get_optim_state_dict(model, optimizers, info)
        _verify_state_dict(model_state_dict, optim_state_dict, info)
        return model_state_dict, optim_state_dict


def _unflatten_model_state_dict(
    model: nn.Module,
    state_dict: dict[nn.Module, dict[str, ValueType]] | dict[str, ValueType],
) -> dict[str, ValueType]:
    if not state_dict:
        return {}

    if isinstance(next(iter(state_dict.keys())), nn.Module):
        warnings.warn(
            "Passing model_state_dict as a ``Dict[nn.Module, Dict[str, Any]]``"
            "is deprecated and will be removed in 2.5. If you need this "
            "feature, please preprocessing the model_state_dict to achieve the "
            "same functionality.",
            FutureWarning,
        )
        cast_state_dict = cast(dict[nn.Module, dict[str, ValueType]], state_dict)
        new_state_dict: dict[str, ValueType] = {}
        for submodule, sub_state_dict in cast_state_dict.items():
            for name, m in model.named_modules():
                if m != submodule:
                    continue

                fqns = _get_fqns(model, name)
                assert len(fqns) == 1, "FQNs for a submodule should only have 1 element"
                prefix = f"{next(iter(fqns))}."
                new_state_dict.update({prefix + subfqn: value for subfqn, value in sub_state_dict.items()})
        return new_state_dict
    else:
        return cast(dict[str, ValueType], state_dict)


def set_model_state_dict(
    model: nn.Module,
    model_state_dict: dict[str, ValueType],
    *,
    options: Optional[StateDictOptions] = None,
) -> _IncompatibleKeys:
    model_state_dict: dict[str, ValueType] = _unflatten_model_state_dict(model, model_state_dict)
    with _gc_context():
        info = _verify_options(model, (), optim_only=False, options=options)

        _verify_state_dict(model_state_dict, {}, info)
        return _load_model_state_dict(model, model_state_dict, info)


def set_optimizer_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    optim_state_dict: OptimizerStateType,
    *,
    options: Optional[StateDictOptions] = None,
) -> None:
    with _gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(model, optimizers, optim_only=True, options=options)

        _verify_state_dict({}, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)


def set_state_dict(
    model: nn.Module,
    optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer],
    *,
    model_state_dict: dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
    options: Optional[StateDictOptions] = None,
) -> _IncompatibleKeys:

    model_state_dict: dict[str, ValueType] = _unflatten_model_state_dict(model, model_state_dict)
    with _gc_context():
        optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
        info = _verify_options(model, optimizers, optim_only=not model_state_dict, options=options)

        _verify_state_dict(model_state_dict, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)
        return _load_model_state_dict(model, model_state_dict, info)


@no_type_check
def _patch_model_state_dict(
    model: nn.Module,
    *,
    options: Optional[StateDictOptions] = None,
) -> None:

    _state_dict_call = functools.partial(
        get_model_state_dict,
        model=model,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()

    model.state_dict = state_dict_call

    _load_state_dict_call = functools.partial(
        set_model_state_dict,
        model=model,
        options=options,
    )

    def load_state_dict_call(state_dict: dict[str, Any]):
        _load_state_dict_call(model_state_dict=state_dict)

    model.load_state_dict = load_state_dict_call

    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)


@no_type_check
def _patch_optimizer_state_dict(
    model: nn.Module,
    *,
    optimizers: tuple[torch.optim.Optimizer, ...],
    options: Optional[StateDictOptions] = None,
) -> None:

    _state_dict_call = functools.partial(
        get_optimizer_state_dict,
        model=model,
        optimizers=optimizers,
        options=options,
    )

    def state_dict_call():
        return _state_dict_call()

    _load_state_dict_call = functools.partial(
        set_optimizer_state_dict,
        model=model,
        optimizers=optimizers,
        options=options,
    )

    def load_state_dict_call(state_dict: dict[str, Any]):
        _load_state_dict_call(optim_state_dict=state_dict)

    _patched_state_dict.add(state_dict_call)
    _patched_state_dict.add(load_state_dict_call)
    optimizers = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
    for optim in optimizers:
        optim.state_dict = state_dict_call
        optim.load_state_dict = load_state_dict_call
