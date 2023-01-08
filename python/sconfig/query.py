from __future__ import annotations

__all__ = ["QueryItem", "Query", "ReQuery", "get_query_items_from_path"]

import re
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from ._logger import get_logger

if TYPE_CHECKING:
    from .config import BaseNode

_LOGGER = get_logger(__name__)


@dataclass(unsafe_hash=True, frozen=True)
class QueryItem:
    name: str = ""
    weight: Optional[int] = None
    path: str = ""
    has_data: bool = False

    @classmethod
    def from_node(cls, node: BaseNode) -> QueryItem:
        """ """
        return cls(
            name=node.name, weight=node.weight, path=node.path, has_data=node.has_data
        )


def get_query_items_from_path(node_path: str) -> List[QueryItem]:
    """ """
    return [
        QueryItem(name=v, weight=k)
        for k, v in enumerate(filter(None, node_path.split("/")))
    ]


@dataclass
class Query:
    items: List[QueryItem] = field(default_factory=list)

    def match(self, node: BaseNode):
        _LOGGER.info(f"exact matching: {self.items} {node!r}")

        node_dict = vars(node)
        _FILTER_EXPR = lambda item: item[-1] and node_dict.get(item[0]) != item[-1]

        for item in self.items:
            item_dict = dict(filter(_FILTER_EXPR, vars(item).items()))
            if not item_dict:
                _LOGGER.debug(f"match: {item} == {node!s}")
                return True

        return False

    @classmethod
    def from_path(cls, node_path: str) -> Query:
        """ """
        return cls(get_query_items_from_path(node_path))


@dataclass
class ReQuery(Query):
    def match(self, node: BaseNode):
        _LOGGER.info(f"regex matching: {self.items} {node!s}")

        for c in self.items:
            if node.weight == c.weight and re.match(c.name, node.name):
                _LOGGER.debug(f"match: {c.name} == {node.name}")
                return True

        return False
