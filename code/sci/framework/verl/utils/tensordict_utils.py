# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Any, Iterable

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack


def assign_non_tensor_data(tensor_dict: TensorDict, key, val):
    assert isinstance(tensor_dict, TensorDict), "input dict must be a TensorDict"
    tensor_dict[key] = NonTensorData(val)


def assign_non_tensor_stack(tensor_dict: TensorDict, key, val: list):
    assert isinstance(tensor_dict, TensorDict), "input dict must be a TensorDict"
    tensor_dict[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])


def assign_non_tensor(tensor_dict: TensorDict, **kwargs):
    assert isinstance(tensor_dict, TensorDict), "input dict must be a TensorDict"
    for key, val in kwargs.items():
        if isinstance(val, (NonTensorData | NonTensorStack)):
            tensor_dict[key] = val
        elif isinstance(val, list):
            assign_non_tensor_stack(tensor_dict=tensor_dict, key=key, val=val)
        else:
            assign_non_tensor_data(tensor_dict=tensor_dict, key=key, val=val)
    return tensor_dict


def unwrap_non_tensor_data(data):
    if isinstance(data, NonTensorData):
        return data.data
    return data


def get_non_tensor_data(data: TensorDict, key: str, default):
    output = data.get(key, default)
    return unwrap_non_tensor_data(output)


def concat_nested_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    for tensor in tensors:
        assert tensor.is_nested and tensor.is_contiguous()
    unbind_tensors = []
    for tensor in tensors:
        assert len(tensor.shape) == 2, f"nested tensor must have 2 dimensions. Got {tensor.shape}"
        unbind_tensor = tensor.unbind(0)
        unbind_tensors.extend(list(unbind_tensor))

    tensor = torch.nested.as_nested_tensor(unbind_tensors, layout=torch.jagged)
    return tensor


def concat_tensordict_with_none_bsz(data: list[TensorDict]):
    for d in data:
        assert len(d.batch_size) == 0
    return data[0]


def concat_tensordict(data: list[TensorDict]) -> TensorDict:
    assert len(data) > 0, "Must have at least one tensordict"

    nested_tensor_keys = {key for key, value in data[0].items() if isinstance(value, torch.Tensor) and value.is_nested}

    if not nested_tensor_keys:
        if len(data[0].batch_size) == 0:
            return concat_tensordict_with_none_bsz(data)
        return TensorDict.cat(data, dim=0)

    regular_tds = []
    for td in data:
        current_nested_keys = {k for k, v in td.items() if isinstance(v, torch.Tensor) and v.is_nested}
        assert current_nested_keys == nested_tensor_keys, "All tensordicts must have the same set of nested tensors."

        regular_items = {k: v for k, v in td.items() if k not in nested_tensor_keys}
        regular_tds.append(TensorDict(regular_items, batch_size=td.batch_size, device=td.device))

    output = TensorDict.cat(regular_tds, dim=0)

    for key in nested_tensor_keys:
        nested_tensors_to_concat = [td[key] for td in data]
        output[key] = concat_nested_tensors(nested_tensors_to_concat)

    return output


def chunk_tensordict(td: TensorDict, chunks: int) -> list[TensorDict]:
    assert isinstance(td, TensorDict) and len(td) % chunks == 0, (
        f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"
    )
    chunk_size = len(td) // chunks
    keys = {key for key, val in td.items() if isinstance(val, torch.Tensor) and val.is_nested and val.dim() >= 3}
    new_td = TensorDict({k: v for k, v in td.items() if k not in keys}, batch_size=td.batch_size, device=td.device)

    tds = new_td.chunk(chunks=chunks)
    for key in keys:
        tensors = td[key].unbind(dim=0)
        for i, td in enumerate(tds):
            td[key] = torch.nested.as_nested_tensor(tensors[i * chunk_size : (i + 1) * chunk_size], layout=torch.jagged)

    return tds


def get_tensordict(tensor_dict: dict[str, torch.Tensor | list], non_tensor_dict: dict = None) -> TensorDict:
    tensor_dict = tensor_dict.copy()
    if non_tensor_dict is None:
        non_tensor_dict = {}

    batch_size = None

    for key, val in tensor_dict.items():
        if isinstance(val, torch.Tensor) and val.is_nested:
            assert val.is_contiguous(), "Nested tensors must be contiguous. Try setting layout=torch.jagged"
            assert val.layout == torch.jagged, "Nested tensors must be jagged."

        if isinstance(val, NonTensorStack):
            if batch_size is None:
                batch_size = len(val)
            else:
                assert len(val) == batch_size, (
                    f"Batch size of NonTensorStack {key} is not consistent with other tensors. "
                    f"Expected {batch_size}, got {len(val)}"
                )
            continue

        if isinstance(val, list):
            for v in val:
                assert not isinstance(v, torch.Tensor), (
                    "Passing a list makes the data NonTensorStack, "
                    "which doesn't support torch.Tensor. Please convert to numpy first"
                )
            tensor_dict[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])

        assert isinstance(val, torch.Tensor | list)

        if batch_size is None:
            batch_size = val.size(0) if isinstance(val, torch.Tensor) else len(val)
        else:
            val_batch_size = val.size(0) if isinstance(val, torch.Tensor) else len(val)
            assert val_batch_size == batch_size, (
                f"Batch size of tensor {key} is not consistent with other tensors. "
                f"Expected {batch_size}, got {val_batch_size}"
            )

    if batch_size is None:
        batch_size = []
    else:
        batch_size = [batch_size]

    for key, val in non_tensor_dict.items():
        assert key not in tensor_dict
        tensor_dict[key] = NonTensorData(val)

    return TensorDict(source=tensor_dict, batch_size=batch_size)


def index_select_tensor_dict(batch: TensorDict, indices: torch.Tensor | list[int]) -> TensorDict:
    if isinstance(indices, list):
        indices = torch.tensor(indices)

    assert indices.dim() == 1, "indices must be a 1D tensor"

    data_dict = {}
    batch_size = indices.shape[0]

    if batch is not None:
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
                data_dict[key] = tensor[indices]
            elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                tensor_lst = tensor.unbind()
                data_dict[key] = torch.nested.as_nested_tensor(
                    [tensor_lst[idx] for idx in indices], layout=torch.jagged
                )
            else:
                if tensor.shape:
                    data_dict[key] = tensor[indices]
                else:
                    data_dict[key] = tensor
        selected_batch = TensorDict(source=data_dict, batch_size=batch_size)
    else:
        selected_batch = None

    return selected_batch


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    )
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2.get(key)
        else:
            if isinstance(tensor_dict2[key], torch.Tensor):
                assert tensor_dict1[key].equal(tensor_dict2[key]), (
                    f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
                )
            else:
                assert tensor_dict1[key] == tensor_dict2[key], (
                    f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
                )

    return tensor_dict1


def make_iterator(tensordict: TensorDict, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
    from torch.utils.data import DataLoader

    assert tensordict.batch_size[0] % mini_batch_size == 0, f"{tensordict.batch_size[0]} % {mini_batch_size} != 0"
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    assert isinstance(dataloader_kwargs, dict)

    idx_lst = torch.arange(tensordict.shape[0])

    train_dataloader = DataLoader(
        dataset=idx_lst, batch_size=mini_batch_size, collate_fn=lambda x: x, generator=generator, **dataloader_kwargs
    )

    def get_data():
        for _ in range(epochs):
            for idx in train_dataloader:
                yield index_select_tensor_dict(tensordict, idx)

    return iter(get_data())


def assert_tensordict_eq(tensordict1: TensorDict, tensordict2: TensorDict):
    tensordict1_key_set = set(tensordict1.keys())
    tensordict2_key_set = set(tensordict2.keys())
    assert tensordict1_key_set == tensordict2_key_set, (
        f"key set diffs. Got {tensordict2_key_set=} vs {tensordict1_key_set=}"
    )

    for key in tensordict1.keys():
        val = tensordict1[key]
        val2 = tensordict2[key]

        assert type(val) is type(val2), f"The type of {key} must be the same. Got {type(val)} vs {type(val2)}"

        if isinstance(val, torch.Tensor):
            if val.is_nested:
                assert val.is_nested and val2.is_nested, (
                    f"Both tensors must be nested tensors. {val.is_nested=}, {val2.is_nested=}"
                )
                t1, t2 = val.unbind(), val2.unbind()
                assert len(t1) == len(t2), f"Nested tensor should have the same lengths. {len(t1)=} vs {len(t2)=}"
                for c1, c2 in zip(t1, t2, strict=True):
                    assert torch.equal(c1, c2), f"Nested tensor components have different values. {c1=} vs {c2=}"
            else:
                assert torch.all(torch.eq(val, val2)).item()
        else:
            assert val == val2


def get(tensordict: TensorDict, key: str, default=None) -> Any:
    if key not in tensordict:
        return default

    output = tensordict.get(key)
    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, NonTensorStack):
        return output.tolist()
    else:
        assert isinstance(output, NonTensorData)
        return output.data


def get_keys(tensordict: TensorDict, keys: Iterable[str]) -> TensorDict:
    tensor_output = {}
    non_tensor_output = {}
    for key in keys:
        if key not in tensordict.keys():
            raise KeyError(f"key {key} not in tensordict")
        output = tensordict.get(key)
        if isinstance(output, torch.Tensor):
            tensor_output[key] = output
        elif isinstance(output, NonTensorStack):
            tensor_output[key] = output.tolist()
        else:
            assert isinstance(output, NonTensorData)
            non_tensor_output[key] = output.data

    return get_tensordict(tensor_output, non_tensor_output)


def pop(tensordict: TensorDict, key: str, default=None) -> Any:
    _sentinel = object()
    output = tensordict.pop(key, _sentinel)
    if output is _sentinel:
        return default

    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, NonTensorStack):
        return output.tolist()
    else:
        assert isinstance(output, NonTensorData)
        return output.data


def pop_keys(tensordict: TensorDict, keys: Iterable[str]) -> TensorDict:
    tensor_output = {}
    non_tensor_output = {}
    for key in keys:
        if key not in tensordict.keys():
            raise KeyError(f"key {key} not in tensordict")
        output = tensordict.get(key)
        if isinstance(output, torch.Tensor):
            tensor_output[key] = tensordict.pop(key)
        elif isinstance(output, NonTensorStack):
            tensor_output[key] = tensordict.pop(key).tolist()
        else:
            assert isinstance(output, NonTensorData)
            non_tensor_output[key] = tensordict.pop(key)

    return get_tensordict(tensor_output, non_tensor_output)


def pad_to_divisor(data: TensorDict, size_divisor: int):
    assert isinstance(data, TensorDict), "data must be a TensorDict"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = torch.cat([data] + padding_protos)
    else:
        if len(data) == 0:
            logging.warning("padding a DataProto with no item, no changed made")
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad(data: TensorDict, pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def contiguous(data: TensorDict) -> TensorDict:
    tensor_dict = {}
    non_tensor_dict = {}

    for key in data.keys():
        val = data.get(key)
        if isinstance(val, NonTensorData):
            non_tensor_dict[key] = val
        elif isinstance(val, NonTensorStack):
            tensor_dict[key] = val
        else:
            assert isinstance(val, torch.Tensor), f"Expect val to be a torch.Tensor. Got {type(val)}"
            tensor_dict[key] = val.contiguous()

    return get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor_dict)


def maybe_fix_3d_position_ids(data: TensorDict):
    if "position_ids" in data.keys() and data["position_ids"].dim() == 3 and data["position_ids"].is_nested:
        data["position_ids"]._ragged_idx = 2
