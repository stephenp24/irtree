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
from typing import TYPE_CHECKING, Any, Deque, Dict, Generator, List, Optional, Set, cast, Iterable

from ._logger import get_logger
from ._profiler import profile
from ._utils import instance_validator
from ._descriptor import (
    StringDescriptor,
    IntDescriptor,
    TotalWeightDescriptor, 
    PathDescriptor, 
    DataItemDescriptor, 
    HasDataDescriptor, 
    ParentDescriptor,
)
from .query import Query, get_query_items_from_path

if TYPE_CHECKING:
    from .query import QueryItem

_LOGGER = get_logger(__name__)


def walk(node: BaseNode) -> Generator[BaseNode, None, None]:
    """DFS-walker"""
    yield node
    children = node.children.copy()
    while children:
        for child in walk(children.popleft()):
            yield child


def get_resolved_data_item(items: Iterable[BaseDataItem]) -> BaseDataItem:
    """ """
    item_type = None
    result = dict()
    for item in items:
        # print(f"@@@@: resolving: {item!r}")
        result.update(item.dict())
        if not item_type:
            item_type = item.__class__

    if not item_type:
        raise RuntimeError(f"Failed to resolve items: {items!s}")

    return item_type(**result)


@dataclass(unsafe_hash=True)
class BaseDataItem:
    author: StringDescriptor = StringDescriptor()
    total_weight: IntDescriptor = IntDescriptor()

    dict = asdict

    def valid(self):
        return any(astuple(self))


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
    #: Node name identifier, this derives the ``path``
    name: StringDescriptor
    #: Node weight, this derives the node position amongst its siblings
    weight: IntDescriptor = IntDescriptor()
    #: Optional data item this node contains
    data_item: DataItemDescriptor = DataItemDescriptor()
    #: Parent node of this instance
    parent: ParentDescriptor = ParentDescriptor()
    #: Sorted deque containing all the child nodes
    children: Deque[BaseNode] = field(default_factory=deque, repr=False)
    #: (read-only) Node total weight, this is the total weight from the root
    total_weight: TotalWeightDescriptor = TotalWeightDescriptor()
    #: (read-only) The node path, this represents the static flattened hierarchy 
    path: PathDescriptor = PathDescriptor()
    #: (read-only) The state whether this node contains any ``data_item`` or not
    has_data: HasDataDescriptor = HasDataDescriptor()

    # --- Not sure if we need ---
    # #: The location of the data item (if stored statically)
    # location: StringDescriptor = StringDescriptor()

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        """load and populate all child"""
        raise NotImplementedError

    # --- Custom member access ---

    @property
    def root(self) -> BaseNode:
        """ (read-only) Get the root node of this node """
        node = self
        while node.parent:
            node = node.parent.root

        return node

    # --- dunder ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseNode):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.path, self.weight, self.data_item))

    def __str__(self):
        """ """
        return f"{self.path} ({self.weight}) ({self.has_data})"

    # --- convenience ---

    def add_child(self, child: BaseNode, safe: bool = True):
        """ Set the given node as this instance's child.
        
        .. note:: The child will be added into the expected index location, sorted 
            based on its weight. 
        
        """
        _LOGGER.debug(f"Adding {child!s}")

        # Note: Local function to make the intention clear on what each
        #       blocks of code is doing.
        def _add_last_child(child):
            self.children.append(child)

        def _add_first_child(child):
            self.children.appendleft(child)

        def _add_mid_child(child, index: int):
            self.children.insert(index, child)

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
                for k, v in groupby(self.children, lambda n: n.weight)
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
            heavier_child_index = self.children.index(heavier_child)

            return heavier_child_index

        # Add the new child, this blocks of code will handle the 
        # sorting logic to find where should the child added in the 
        # list of children
        if child not in self.children:
            # ensure child is of the same type
            with instance_validator(child, self.__class__):
                if not self.children:
                    _LOGGER.debug("\tAdding new child")
                    _add_first_child(child)
                elif not safe:
                    _LOGGER.debug("\tAppending child")
                    _add_last_child(child)
                # Add to the corresponding index based on children's weights
                else:
                    first_child = self.children[0]
                    last_child = self.children[-1]
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
        if child.parent != self:
            child.parent = self

    # Added for consistency because we support `add_child` above
    def remove_child(self, child: BaseNode):
        """ Remove the given node from this instance's list of children. """
        if child in self.children:
            self.children.remove(child)

    def find_child(self, query: Query) -> Optional[BaseNode]:
        for child in self.children:
            if query.match(child):
                return child
        
        return None

    def iter(self, query: Optional[Query] = None) -> Generator[BaseNode, None, None]:
        """ 
        
        Examples:

        >>> # Find child using string comp
        >>> children = node.find_children(query=Query([QueryItem(name="foo")]))
        >>> assert all([n.name == "foo" for n in children])
        >>> 
        >>> # Find child using regex pattern 
        >>> regex_children = node.find_children(query=ReQuery([QueryItem(name="^fo")]))
        >>> assert all([n.name.startswith("fo") for n in regex_children])
        """
        for node in walk(self):
            if query and not query.match(node):
                continue
            yield node

    def get_contributing_nodes(self, query: Query) -> List[BaseNode]:
        """ """
        result = []
        
        if not query.match(self):
            return result

        result.append(self)
        children = self.children.copy()
        while children:
            child = children.popleft()
            result += child.get_contributing_nodes(query=query)

        return sorted(result, key=lambda item: item.total_weight)

    # --- utility ---

    def render(self, format="term") -> None:
        """ """
        prefix = (self.parent.weight + 1) * " " if self.parent else ""
        prefix += "|" + self.weight * "-"
        print(f"{prefix}{self!s}")

        children = self.children.copy()
        while children:
            children.popleft().render(format=format)

    # --- constructor ---

    @classmethod
    def create_from_path(cls, node_path: str) -> BaseNode:
        """Create node from the given unix-like path. The resulting node are the child-most node.
        e.g: "/foo/bar/baz" will return the "baz" node with parent pointing to the "bar" node.
        """
        query_items = deque(get_query_items_from_path(node_path))  # type: Deque[QueryItem]
        root = None
        pnode = None
        while query_items:
            component = query_items.popleft()
            node = cls(component.name, component.weight)
            if pnode:
                pnode.add_child(node)
            else:
                root = node
            pnode = node

        assert root is not None, f"Fail to instantiate from path, can't find parent node: {node_path}"

        return root


@dataclass(unsafe_hash=False, eq=False)
class ConfigNode(BaseNode):
    """Node that handle config item"""

    def load(self):
        """ """
        return BaseDataItem(
            author=f"{self.path} ({self.has_data})",
            total_weight=self.total_weight,
        )

    def get_configs(
        self, query: Optional[Query] = None
    ) -> Generator[BaseDataItem, None, None]:
        """ """
        if query:
            nodes = self.get_contributing_nodes(query=query)
        else:
            nodes = sorted(self.iter(), key=lambda node: node.total_weight)
        
        for node in nodes:
            node.load()
            if node.has_data:
                yield node.data_item

    def get_resolved_config(self, path: Optional[str] = None) -> BaseDataItem:
        """ """
        query = Query(get_query_items_from_path(path)) if path else None
        return get_resolved_data_item(items=self.get_configs(query=query))
