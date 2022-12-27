""" Collection of utility functions that are used across this lib """
from __future__ import annotations  # type hint circular import

__all__ = ["instance_validator", "get_resolved_item", "split_node_path"]

import contextlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Iterable, Type

if TYPE_CHECKING:
    from .config import BaseDataItem, BaseNode


@contextlib.contextmanager
def instance_validator(instance: Any, instance_type: Type):
    if not isinstance(instance, instance_type):
        raise TypeError(f"Invalid instance type: {instance!s} {instance_type}")
    try:
        yield
    finally:
        return


def get_resolved_item(items: Iterable[BaseDataItem]) -> BaseDataItem:
    """ """
    item_type = None
    result = dict()
    for item in items:
        result.update(item.dict())
        if not item_type:
            item_type = item.__class__

    if not item_type:
        raise RuntimeError(f"Failed to resolve items: {items!s}")

    return item_type(**result)


def split_node_path(node_path: str) -> OrderedDict[int, str]:
    """ """
    result = OrderedDict()  # type: OrderedDict[int, str]
    [result.setdefault(k, v) for k, v in enumerate(filter(None, node_path.split("/")))]

    return result
