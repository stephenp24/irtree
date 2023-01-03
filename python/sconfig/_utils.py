""" Collection of utility functions that are used across this lib """
from __future__ import annotations  # type hint circular import

__all__ = ["instance_validator"]

import contextlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Iterable, Type

if TYPE_CHECKING:
    from .config import BaseDataItem, BaseNode


@contextlib.contextmanager
def instance_validator(instance: Any, instance_type: Type):
    if not isinstance(instance, instance_type):
        raise TypeError(f"Invalid type: {instance!s}\nExpected: {instance_type}")
    try:
        yield
    finally:
        return
