""" Self-sorting weighted tree is used to optimise the search.
Time complexity: O(n log n)
"""
from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections import deque
from copy import copy
from dataclasses import asdict, dataclass, field, astuple
from itertools import groupby
from typing import TYPE_CHECKING, Any, Deque, Dict, Generator, List, Optional, Set, cast

from ._logger import get_logger
from ._profiler import profile
from ._utils import get_resolved_item, instance_validator
from .query import BaseQuery

if TYPE_CHECKING:
    from .query import Component

_LOGGER = get_logger(__name__)


@dataclass(unsafe_hash=True)
class BaseDataItem:
    author: str = ""
    sum_weight: int = 0

    dict = asdict

    def valid(self):
        return any(astuple(self))


class DataItemDescriptor:

    # https://docs.python.org/3/whatsnew/3.6.html
    # https://www.python.org/dev/peps/pep-0487/
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, type):
        """
        Args:
            obj: The instance of the object your descriptor is attached to.
            type: The type of the object the descriptor is attached to.
        """
        if obj:
            return obj.__dict__.get(self._name) or None
        return None

    def __set__(self, obj, value: BaseDataItem):
        if value is None:
            obj.__dict__[self._name] = value
        else:
            with instance_validator(value, BaseDataItem):
                if value.valid():
                    obj.__dict__[self._name] = value


class ParentDescriptor:
    """ Use weakref to avoid memory leaks when child references the parent so 
    that on references reset, together with the root deletes all descendants 
    cascading. """

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, type):
        """ """
        if obj:
            parent = obj.__dict__.get(self._name)
            if parent:
                return parent()
        return None

    def __set__(self, obj, value):
        if value is None:
            obj.__dict__[self._name] = None
        else:
            with instance_validator(value, BaseNode):
                obj.__dict__[self._name] = weakref.ref(value)


# As per docs:
# Here are the rules governing implicit creation of a __hash__() method.
# Note that you cannot both have an explicit __hash__() method in your dataclass
# and set unsafe_hash=True; this will result in a TypeError.
#
# If eq is true and frozen is false, __hash__() will be set to None,
# marking it unhashable (which it is, since it is mutable)
#
@dataclass(unsafe_hash=False, eq=False)
class BaseNode:
    """ Node must have at least a name and weight """
    name: str
    weight: int
    location: Optional[str] = None
    data_item: DataItemDescriptor = DataItemDescriptor()
    parent: ParentDescriptor = ParentDescriptor()
    __children: Deque[BaseNode] = field(default_factory=deque)

    def __post_init__(self):
        """ dataclass """
        self.__update_path()

    # --- Custom member access ---
    @property
    def has_data(self) -> bool:
        return self.data_item is not None
    
    @property
    def path(self) -> str:
        """ Get the node path. This is a read-only """
        return f"{self.__parent.path if self.__parent else ''}/{self.name}"

    def __update_path(self):
        """ """
        self.__path = f"{self.__parent.path if self.__parent else ''}/{self.name}"

    # def get_children(self) -> Deque[BaseNode]:
        """ Get the children sorted based on its weight. 
        
        .. important:: this returns a copy of this node's children. To 
            modify the children, consider using `add_child` or `remove_child`.

        """
        return self.__children.copy()

    def add_child(self, child: BaseNode, safe: bool = True):
        """ Set the given node as this instance's child.
        
        .. note:: The child will be added into the expected index location, sorted 
            based on its weight. 
        
        """
        _LOGGER.debug(f"Adding {child!s}")

        # Note: Local function to make the intention clear on what each
        #       blocks of code is doing.
        def _add_last_child(child):
            self.__children.append(child)

        def _add_first_child(child):
            self.__children.appendleft(child)

        def _add_mid_child(child, index: int):
            self.__children.insert(index, child)

        def _get_mid_child_index(child) -> int:
            """ Get the ``mid-child index``.
            ``Mid-child index`` is the current index of the first found child 
            with weight heavier than the new child from the current list of 
            children.
            """
            # Group the current children based on its weight to help us find
            # the next-heavier child item by comparing its weight
            weight_to_children = {
                k: deque(v)
                for k, v in groupby(self.__children, lambda n: n.weight)
            }  # type: Dict[int, Deque[BaseNode]]

            # Get the index of the heavier weight
            sorted_weights = sorted(weight_to_children.keys())
            if child.weight in weight_to_children:
                _LOGGER.debug(f"\tExising weight")
                heavier_weight_index = sorted_weights.index(child.weight) + 1
            else:
                _LOGGER.debug(f"\tNew weight")
                # Note: This shouldn't raise any StopIteration exception because 
                #       at this point, we know for sure the new child weight is 
                #       between than the first and last child's weight.
                heavier_weight = next(
                    filter(lambda i: i > child.weight, sorted_weights)
                )
                heavier_weight_index = sorted_weights.index(heavier_weight)

            # Get the heavier weight child instance from the groups
            heavier_child_weight = sorted_weights[heavier_weight_index]
            heavier_child = weight_to_children.pop(heavier_child_weight).popleft()
            heavier_child_index = self.__children.index(heavier_child)

            return heavier_child_index

        # Make sure child's parent point to this node
        if child not in self.__children:
            # Always sort as we add.
            with instance_validator(child, self.__class__):
                if not self.__children:
                    _LOGGER.debug("\tAdding new child")
                    _add_first_child(child)
                # Add the child as-is
                elif not safe:
                    _LOGGER.debug("\tAppending child")
                    _add_last_child(child)
                # Add to the corresponding index based on children's weights
                else:
                    first_child = self.__children[0]
                    last_child = self.__children[-1]
                    if first_child.weight > child.weight:
                        _LOGGER.debug("\tAdding first child")
                        _add_first_child(child)
                    elif last_child.weight <= child.weight:
                        _LOGGER.debug("\tAdding last child")
                        _add_last_child(child)
                    else:
                        # Steps to add middle child:
                        # 1. Find the next heavier child
                        # 2. Insert the new child on the heavier child index
                        _LOGGER.debug("\tAdding middle child")
                        mid_child_index = _get_mid_child_index(child)
                        _LOGGER.debug(f"\tMid child index: {mid_child_index}")
                        _add_mid_child(child, mid_child_index)
        
        # Ensure the new child's parent points to this node.
        child.__parent = self
        child.__update_path()

    def remove_child(self, child: BaseNode):
        """ Remove the given node from this instance's list of children. """
        if child in self.__children:
            self.__children.remove(child)

    # def get_parent(self) -> Optional[BaseNode]:
    #     return self.__parent

    # def set_parent(self, parent: BaseNode):
        with instance_validator(parent, self.__class__):
            parent.add_child(self)

        return

    def get_root(self) -> BaseNode:
        """ """
        node = self
        while node.get_parent():
            node = cast(BaseNode, node.get_parent()).get_root()

        return node

    # --- dunder ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseNode):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        d = vars(self).copy()
        # When hashing, we strip the children, this should be fine because
        # there shouldn't be two identical node that has the same parent.
        # READ: each child of parent is unique
        key = "_BaseNode__children"
        if key in d:
            del d[key]
        return hash(tuple(d.items()))

    def __str__(self):
        """ """
        return f"{self.name} ({self.weight}) ({self.has_data})"

    # --- convenience ---

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        """load and populate all child"""
        raise NotImplementedError

    def render(self) -> None:
        """ """
        parent = self.get_parent()
        prefix = (parent.weight + 1) * " " if parent else ""
        prefix += "|" + self.weight * "-"
        print(f"{prefix}{self!s}")

        children = self.get_children()
        while children:
            children.popleft().render()

    def get_sum_weight(self) -> int:
        """ """
        parent_weight = (
            cast(BaseNode, self.get_parent()).get_sum_weight()
            if self.get_parent()
            else 0
        )
        return self.weight + parent_weight

    def get_all_data(self, query: Optional[BaseQuery] = None) -> List[BaseDataItem]:
        """ """
        result = []
        query_match = query.match(self) if query else True
        if query_match:
            self.load()
            if self.has_data:
                _LOGGER.debug(f"{self!s}")
                result.append(cast(BaseDataItem, self.get_data_item()))

            children = self.get_children()
            while children:
                result += children.popleft().get_all_data(query=query)

            result = sorted(result, key=lambda item: item.sum_weight)

        return result

    # --- constructor ---

    @classmethod
    def create_from_path(cls, node_path: str) -> BaseNode:
        """Create node from the given unix-like path. The resulting node are the child-most node.
        e.g: "/foo/bar/baz" will return the "baz" node with parent pointing to the "bar" node.
        """
        components = deque(
            BaseQuery.from_node_path(node_path).components
        )  # type: Deque[Component]
        pnode = None
        while components:
            component = components.popleft()
            node = cls(name=component.value, weight=component.weight)
            if pnode:
                pnode.add_child(node)
            pnode = node

        assert pnode is not None

        return pnode


@dataclass(unsafe_hash=False, eq=False)
class ConfigNode(BaseNode):
    """Node that handle config item"""

    def load(self):
        """ """
        return BaseDataItem(
            author=f"{self.get_parent()!s}/{self!s}",
            sum_weight=self.get_sum_weight(),
        )

    def get_configs(
        self, query: Optional[BaseQuery] = None
    ) -> Generator[BaseDataItem, None, None]:
        """ """
        for data in self.get_all_data(query=query):
            yield data

    def get_resolved_config(self, query: Optional[BaseQuery] = None) -> BaseDataItem:
        """ """
        return get_resolved_item(items=self.get_configs(query=query))


def walk(node: BaseNode) -> Generator:
    """DFS-walker"""
    yield node
    children = node.get_children()
    while children:
        for child in walk(children.popleft()):
            yield child
