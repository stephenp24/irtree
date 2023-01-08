from __future__ import annotations

__all__ = [
    "StringDescriptor",
    "IntDescriptor",
    "TotalWeightDescriptor",
    "PathDescriptor",
    "DataItemDescriptor",
    "HasDataDescriptor",
    "ParentDescriptor",
]

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, cast

from six import string_types

if TYPE_CHECKING:
    from irtree.node import BaseDataItem, BaseNode


# https://docs.python.org/3/whatsnew/3.6.html
# https://www.python.org/dev/peps/pep-0487/
class Validator(ABC):
    _DEFAULT = None  # type: Union[None, str, int, BaseNode]

    def __init__(self, default=None):
        if default is not None:
            self._DEFAULT = default

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj:
            result = obj.__dict__.get(self._name) or self._DEFAULT
        else:
            result = self._DEFAULT
        return result

    def __set__(self, obj, value):
        self.validate(obj, value)
        obj.__dict__[self._name] = value

    @abstractmethod
    def validate(self, obj, value):
        pass


class ReadOnlyValidator(Validator):
    def __get__(self, obj, objtype=None):
        result = super(ReadOnlyValidator, self).__get__(obj, objtype)
        if obj:
            obj.__dict__[self._name] = result
        return result

    def __set__(self, obj, value):
        if value:
            raise AttributeError(f"{self._name!r} is read-only.")


class StringDescriptor(Validator):
    _DEFAULT = ""

    def __get__(self, obj, objtype=None) -> str:
        return super(StringDescriptor, self).__get__(obj, objtype)

    def validate(self, obj, value):
        if not value:
            raise TypeError(f"`{obj.__class__.__name__}` missing 1 required positional argument: 'name'")
        if (value is not None) and (not isinstance(value, string_types)):
            raise TypeError(f"Expected {value!r} to be a string")


class IntDescriptor(Validator):
    _DEFAULT = 0

    def __get__(self, obj, objtype=None) -> int:
        return super(IntDescriptor, self).__get__(obj, objtype)

    def validate(self, obj, value):
        if (value is not None) and (not isinstance(value, int)):
            raise TypeError(f"Expected {value!r} to be an int")


class PathDescriptor(ReadOnlyValidator, StringDescriptor):
    """${parent.path}/${name}"""

    def __get__(self, obj: BaseNode, objtype=None) -> str:
        from .node import _root

        if obj:
            path = f"{obj.parent.path if not isinstance(obj.parent, _root) else ''}/{obj.name}"
            obj.__dict__[self._name] = path

        return super(PathDescriptor, self).__get__(obj, objtype)


class TotalWeightDescriptor(ReadOnlyValidator, IntDescriptor):
    """${parent.weight} + ${weight}"""

    def __get__(self, obj, objtype=None) -> int:
        if obj:
            total_weight = obj.weight
            total_weight += obj.parent.total_weight if obj.parent else 0
            obj.__dict__[self._name] = total_weight

        return super(TotalWeightDescriptor, self).__get__(obj, objtype)


class HasDataDescriptor(ReadOnlyValidator):
    """${data_item} is not None and ${data_item.valid()}"""

    _DEFAULT = False

    def __get__(self, obj: BaseNode, objtype=None) -> bool:
        if obj:
            has_data = obj.data_item and obj.data_item.valid()
            obj.__dict__[self._name] = has_data

        return super(HasDataDescriptor, self).__get__(obj, objtype)

    def validate(self, obj, value):
        pass


class DataItemDescriptor(Validator):
    def __get__(self, obj: BaseNode, objtype=None) -> BaseDataItem:
        return super(DataItemDescriptor, self).__get__(obj, objtype)

    def validate(self, obj, value: BaseDataItem):
        from .node import BaseDataItem

        if value is not None:
            if not isinstance(value, BaseDataItem):
                raise TypeError(
                    f"Expected {value!r} to be a ``BaseDataItem`` typed class"
                )
            elif not value.valid():
                raise ValueError(
                    f"Expected {value!r} to be a valid Item with at least one non-empty attributes"
                )


class ParentDescriptor(Validator):
    """Use weakref to avoid memory leaks since we're referencing both directions
    for each parent/ child node. When child references the parent so
    that on references reset, together with the root deletes all descendants
    cascading."""
    _DEFAULT = None  # type: BaseNode

    def __get__(self, obj: BaseNode, objtype=None) -> BaseNode:
        """ """
        if obj:
            parent = obj.__dict__.get(self._name)
            if parent:
                parent = parent()
            return parent
        return self._DEFAULT

    def __set__(self, obj: BaseNode, value: BaseNode):
        """Set the obj parent ot point to the given node value"""
        self.validate(obj, value)
        obj.__dict__[self._name] = value if value is None else weakref.ref(value)
        # Add this node to the parent's list of children
        if value:
            value.add_child(obj)

    def validate(self, obj, value: BaseNode):
        from .node import BaseNode, _root

        if (value is not None) and (not isinstance(value, (BaseNode, _root))):
            raise TypeError(f"Expected {value!r} to be a ``BaseNode`` typed class")
