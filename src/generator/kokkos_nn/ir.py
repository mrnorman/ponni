from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any

import numpy as np


class DType(str, Enum):
    BOOL = "bool"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"


@dataclass(frozen=True)
class Symbol:
    name: str


Dimension = int | Symbol


@dataclass
class ConstantTensor:
    name: str
    shape: tuple[int, ...]
    dtype: DType
    values: np.ndarray = field(repr=False)
    canonical_layout: str = "onnx"
    learned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "canonical_layout": self.canonical_layout,
            "learned": self.learned,
        }


@dataclass
class TensorValue:
    id: int
    name: str
    shape: tuple[Dimension, ...]
    dtype: DType
    producer: int | None = None
    consumers: list[int] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    is_constant: bool = False
    constant_name: str | None = None

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(dim for dim in self.shape if isinstance(dim, int))

    @property
    def sample_size(self) -> int:
        result = 1
        for dim in self.sample_shape:
            result *= dim
        return result

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dtype"] = self.dtype.value
        data["shape"] = [dim.name if isinstance(dim, Symbol) else dim for dim in self.shape]
        return data


@dataclass
class Node:
    id: int
    op: str
    inputs: list[int]
    outputs: list[int]
    attributes: dict[str, Any] = field(default_factory=dict)
    source_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Graph:
    inputs: list[int]
    outputs: list[int]
    tensors: dict[int, TensorValue]
    nodes: list[Node]
    constants: dict[str, ConstantTensor]
    metadata: dict[str, Any] = field(default_factory=dict)

    def rebuild_links(self) -> None:
        for tensor in self.tensors.values():
            tensor.producer = None
            tensor.consumers = []
        for node in self.nodes:
            for tensor_id in node.inputs:
                self.tensors[tensor_id].consumers.append(node.id)
            for tensor_id in node.outputs:
                self.tensors[tensor_id].producer = node.id

    def renumber_nodes(self) -> None:
        for node_id, node in enumerate(self.nodes):
            node.id = node_id
        self.rebuild_links()

    def node_by_id(self, node_id: int) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tensors": {str(key): value.to_dict() for key, value in sorted(self.tensors.items())},
            "nodes": [node.to_dict() for node in self.nodes],
            "constants": {key: value.to_dict() for key, value in sorted(self.constants.items())},
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def clone(self) -> Graph:
        import copy

        return copy.deepcopy(self)
