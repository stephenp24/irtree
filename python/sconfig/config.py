""" Self-sorting weighted tree is used to optimise the search.
Time complexity: O(n log n)
"""
from __future__ import annotations

# import rapidjson as json
import json
import sys
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from dataclasses import asdict, astuple, dataclass, field, make_dataclass
from enum import Enum, auto
from importlib import import_module
from itertools import groupby
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    TypeVar,
    cast,
)

from sconfig._descriptor import (
    DataItemDescriptor,
    HasDataDescriptor,
    IntDescriptor,
    ParentDescriptor,
    PathDescriptor,
    StringDescriptor,
    TotalWeightDescriptor,
)
from sconfig._logger import get_logger
from sconfig._utils import import_string, instance_validator
from sconfig.query import Query, QueryItem, get_query_items_from_path

_LOGGER = get_logger(__name__)


__KLS_NAME__ = "__kls_name__"


class ITER_MODE(Enum):
    DFS = auto()
    BFS = auto()


DEFAULT_ITER = ITER_MODE.DFS


def walk(
    node: BaseNode, mode: ITER_MODE = ITER_MODE.DFS
) -> Generator[BaseNode, None, None]:
    """simple node walker"""

    def _dfs(node):
        yield node

        children = deque(node.children)
        while children:
            yield from _dfs(children.popleft())

    def _bfs(node):
        visited = deque([node])
        queue = deque(visited)
        while queue:
            child = queue.popleft()
            yield child

            for sibling in child.children:
                if sibling not in visited:
                    visited.append(sibling)
                    queue.append(sibling)

    mode = mode or DEFAULT_ITER
    if ITER_MODE(mode) == ITER_MODE.DFS:
        iter_fnc = _dfs
    else:
        iter_fnc = _bfs

    yield from iter_fnc(node)


def to_term(node: BaseNode):
    """ """

    def pretty_str(node):
        prefix = (node.parent.weight + 1) * " " if node.parent else ""
        prefix += "|" + node.weight * "-"
        return f"{prefix}{node}"

    result = [pretty_str(node) for node in walk(node)]
    result = "\n".join(result)
    return result


def to_json(node: BaseNode):
    """Simple node to json writer"""
    fullname = lambda o: f"{o.__class__.__module__}.{o.__class__.__name__}"

    def serialize(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, deque):
            return tuple(obj)
        elif isinstance(obj, BaseNode):
            d = {__KLS_NAME__: fullname(obj)}
            d.update(deepcopy(vars(obj)))
            for k in ("parent", "path", "has_data", "total_weight"):
                if k in d:
                    del d[k]
            if "children" in d:
                d["children"] = tuple(obj.children)
            return d
        elif isinstance(obj, BaseDataItem):
            d = {__KLS_NAME__: fullname(obj)}
            d.update(deepcopy(vars(obj)))
            return d

    return json.loads(json.dumps(node, default=serialize, indent=4))


def load_json(input_path: str) -> BaseNode:
    """Simple json to node reader"""

    def hook(d):
        if not __KLS_NAME__ in d:
            return

        kls = import_string(d.pop(__KLS_NAME__))
        try:
            res = kls(d["name"], weight=d["weight"])
            res.children = Children()
            for child in d["children"]:
                res.add_child(child)
            if "data_item" in d:
                res.data_item = d["data_item"]
        except KeyError:
            res = kls(**d)
        return res

    with open(input_path, mode="r") as buf:
        node = json.load(buf, object_hook=hook)

    return node


class RENDER_MODE(Enum):
    TERM = to_term
    JSON = to_json


def get_resolved_data_item(items: Iterable[BaseDataItem]) -> BaseDataItem:
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


@dataclass(eq=False)
class BaseDataItem:
    author: StringDescriptor = StringDescriptor()
    total_weight: IntDescriptor = IntDescriptor()

    dict = asdict

    def __eq__(self, other):
        if not isinstance(other, BaseDataItem):
            return False
        return self.dict() == other.dict()

    def valid(self):
        return any(astuple(self))


@dataclass(eq=False)
class Children:
    """Utility iterable class for children representation.

    Characteristic:
    - Iterable
    - Sorted (based on weights and insertion order)
    """

    # Sorted child keys
    names: Deque = field(default_factory=deque, init=False)
    # Unsorted children maps
    children: Dict = field(default_factory=dict, repr=False)

    def __iter__(self):
        """ """
        for key in self.names:
            yield self.children[key]

    def __contains__(self, child):
        """ """
        contain = child.name in self.names
        return contain

    def __eq__(self, other):
        """ """
        if not isinstance(other, Children):
            return False
        return self.children.values() == other.children.values()

    def __len__(self) -> int:
        """ """
        return len(self.names)

    def flush(self):
        self.names = deque()
        self.children = dict()

    def add(self, child, safe: bool = True):
        """ """

        """ Set the given node as this instance's child.
        
        .. note:: The child will be added into the expected index location, sorted 
            based on its weight. 
        
        """
        # Note: Local function to make the intention clear on what each
        #       blocks of code is doing.
        def _add_last_child(child):
            self.names.append(child.name)

        def _add_first_child(child):
            self.names.appendleft(child.name)

        def _add_mid_child(child, index: int):
            self.names.insert(index, child.name)

        def _get_mid_child_index(child) -> int:
            """Get the ``mid-child index``.
            ``Mid-child index`` is the current index of the first found child
            with weight heavier than the new child from the current list of
            children.
            """
            # Group the current children based on its weight to help us find
            # the next-heavier child item by comparing its weight
            weight_to_children = {
                k: deque(v) for k, v in groupby(self, lambda n: n.weight)
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
            heavier_child_index = self.names.index(heavier_child.name)

            return heavier_child_index

        # Add the new child, this blocks of code will handle the
        # sorting logic to find where should the child added in the
        # list of children
        if not self.children:
            _LOGGER.debug("\tAdding new child")
            _add_first_child(child)
        elif not safe:
            _LOGGER.debug("\tAppending child")
            _add_last_child(child)
        elif child not in self:
            # # ensure child is of the same type
            # with instance_validator(child, self.__class__):
            # Add to the corresponding index based on children's weights
            first_child = self.children[self.names[0]]
            last_child = self.children[self.names[-1]]
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
        else:
            return child

        self.children[child.name] = child
        return child

    def remove(self, child):
        if child in self:
            self.names.remove(child.name)
            del self.children[child.name]


# Pseudoroot that kept the reference to all the created nodes
# This helps maintain the node references and flush them from
# memory (allowing the floating nodes to be garbage-collected)
# anytime the user consider to do so.
_root = make_dataclass(
    "_root",
    [
        ("name", str, field(default="root", repr=False)),
        ("path", str, "/"),
        ("weight", int, field(default=0, repr=False)),
        ("total_weight", int, field(default=0, repr=False)),
        ("children", Children, field(default_factory=Children, init=False)),
    ],
    frozen=True,
    namespace={
        "add_child": lambda self, child: self.children.add(child),
        "flush": lambda self: self.children.flush(),
    },
)
ROOT = _root()


@dataclass(eq=False)
class BaseNode:
    """Node must have at least a name and weight"""

    #: Node name identifier, this derives the ``path``
    name: StringDescriptor
    #: Node weight, this derives the node position amongst its siblings
    weight: IntDescriptor = IntDescriptor()
    #: Optional data item this node contains
    data_item: DataItemDescriptor = DataItemDescriptor()
    #: Parent node of this instance
    parent: ParentDescriptor = ParentDescriptor(default=ROOT)
    #: Sorted deque containing all the child nodes
    children: Children = field(default_factory=Children)
    #: (read-only) Node total weight, this is the total weight from the root
    total_weight: TotalWeightDescriptor = TotalWeightDescriptor()
    #: (read-only) The node path, this represents the static flattened hierarchy
    path: PathDescriptor = PathDescriptor()
    #: (read-only) The state whether this node contains any ``data_item`` or not
    has_data: HasDataDescriptor = HasDataDescriptor()

    # --- Not sure if we need ---
    # #: The location of the data item (if stored statically)
    # location: StringDescriptor = StringDescriptor()
    __VERSION__: int = field(
        default=0, init=False, repr=False, hash=False, compare=False
    )

    # --- Custom member access ---

    @property
    def top_node(self) -> BaseNode:
        """(read-only) Get the root node of this node"""
        node = self
        while node.parent != ROOT:
            node = node.parent.top_node

        return node

    # --- dunder ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseNode):
            return False
        return all(
            [
                self.path == other.path,
                self.weight == other.weight,
                self.data_item == other.data_item,
            ]
        )

    def __str__(self):
        """ """
        return f"{self.name} ({self.weight}) ({self.has_data})"

    def __add__(self, other: BaseNode):
        return self.union(other)

    def __enter__(self) -> BaseNode:
        """ """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    # --- convenience ---

    def add_child(self, child: BaseNode, safe: bool = True):
        """Set the given node as this instance's child.

        .. note:: The child will be added into the expected index location, sorted
            based on its weight.

        """
        _LOGGER.debug(f"Adding {child!s}")
        with instance_validator(child, self.__class__):
            child = self.children.add(child, safe=safe)
            # Ensure the new child's parent points to this node.
            if child.parent != self:
                child.parent = self

        return child

    # Added for consistency because we support `add_child` above
    def remove_child(self, child: BaseNode):
        """Remove the given node from this instance's list of children."""
        self.children.remove(child)

    def find_child(self, query: Query) -> Optional[BaseNode]:
        for child in self.children:
            if query.match(child):
                return child

        return None

    def has_child(self, child: BaseNode) -> bool:
        return child in self.children

    # --- iterator ---

    def iter(
        self, query: Optional[Query] = None, mode: ITER_MODE = ITER_MODE.DFS
    ) -> Generator[BaseNode, None, None]:
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
        mode = ITER_MODE(mode) or DEFAULT_ITER
        for node in walk(self, mode=mode):
            if query and not query.match(node):
                continue
            yield node

    def get_contributing_nodes(self, query: Query) -> List[BaseNode]:
        """ """
        result = []

        if not query.match(self):
            return result

        result.append(self)
        children = deque(self.children)
        while children:
            child = children.popleft()
            result += child.get_contributing_nodes(query=query)

        return sorted(result, key=lambda item: item.total_weight)

    def iter_data_items(
        self, query: Optional[Query] = None
    ) -> Generator[BaseDataItem, None, None]:
        """ """
        if query:
            nodes = self.get_contributing_nodes(query=query)
        else:
            nodes = sorted(self.iter(), key=lambda node: node.total_weight)

        for node in nodes:
            if node.has_data:
                yield node.data_item

    def get_resolved_config(self, path: Optional[str] = None) -> BaseDataItem:
        """ """
        query = Query.from_path(path) if path else None
        return get_resolved_data_item(items=self.iter_data_items(query=query))

    # --- utility ---

    def render(self, render_mode: RENDER_MODE = RENDER_MODE.TERM) -> None:
        """ """
        render_mode = render_mode or RENDER_MODE.TERM
        return render_mode(self)

    def write(self, output_path: str, render_mode: RENDER_MODE = RENDER_MODE.JSON):
        """ """
        render_mode = render_mode or RENDER_MODE.JSON
        out = render_mode(self)
        with open(output_path, mode="w") as buf:
            if render_mode == RENDER_MODE.JSON:
                json.dump(out, buf, indent=4)
            elif render_mode == RENDER_MODE.TERM:
                buf.write(out)

    def update(self, other: BaseNode):
        """ """
        raise NotImplementedError
        # if not isinstance(other, self.__class__):
        #     raise TypeError(f"Invalid type: {type(other)} expect {self.__class__!r}")

    def union(self, other: BaseNode) -> BaseNode:
        """ """
        raise NotImplementedError
        # result = deepcopy(self)
        # result.update(other)

        # return result

    # --- constructor ---

    @classmethod
    def create_from_path(cls, node_path: str) -> BaseNode:
        """Create node from the given unix-like path. The resulting node are the child-most node.
        e.g: "/foo/bar/baz" will return the "baz" node with parent pointing to the "bar" node.
        """
        query_items = deque(
            get_query_items_from_path(node_path)
        )  # type: Deque[QueryItem]
        root = None
        pnode = None
        while query_items:
            query_item = query_items.popleft()
            node = cls(query_item.name, query_item.weight)
            if pnode:
                pnode.add_child(node)
            else:
                root = node
            pnode = node

        assert (
            root is not None
        ), f"Fail to instantiate from path, can't find parent node: {node_path}"

        return root


@dataclass(eq=False)
class ContextualNode(BaseNode):
    """Node that handle config item"""

    # --- dunder ---

    def __eq__(self, other: object) -> bool:
        return super(ContextualNode, self).__eq__(other)

    # --- constructor ---

    @classmethod
    def create_from_context(
        cls, ctx: Dict, weight_maps: Dict[str, int]
    ) -> ContextualNode:
        node = None
        for k, v in ctx.items():
            _node = cls(v, weight=weight_maps[k])
            if node:
                node.add_child(_node)
            node = _node

        return node.top_node
