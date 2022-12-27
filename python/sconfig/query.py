from __future__ import annotations

__all__ = ["Component", "BaseQuery", "ExactQuery", "RegexQuery"]

import re
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from ._logger import get_logger
from ._utils import split_node_path

if TYPE_CHECKING:
    from .config import BaseNode

_LOGGER = get_logger(__name__)


@dataclass(unsafe_hash=True, frozen=True, repr=False)
class Component:
    weight: int = 0
    value: str = ""

    def __repr__(self):
        return f"C({self.weight}, {self.value})"


@dataclass
class BaseQuery:
    components: List[Component] = field(default_factory=list)

    @abstractmethod
    def match(self, node: BaseNode):
        """ """
        raise NotImplementedError()

    @classmethod
    def from_node_path(cls, node_path: str) -> BaseQuery:
        """ """
        return cls([Component(k, v) for k, v in split_node_path(node_path).items()])


@dataclass
class ExactQuery(BaseQuery):
    def match(self, node: BaseNode):
        _LOGGER.info(f"exact matching: {self.components} {node!s}")

        for c in self.components:
            if node.weight == c.weight and node.name == c.value:
                _LOGGER.debug(f"match: {c.value} == {node.name}")
                return True

        return False


@dataclass
class RegexQuery(BaseQuery):
    def match(self, node: BaseNode):
        _LOGGER.info(f"regex matching: {self.components} {node!s}")

        for c in self.components:
            if node.weight == c.weight and re.match(c.value, node.name):
                _LOGGER.debug(f"match: {c.value} == {node.name}")
                return True

        return False
