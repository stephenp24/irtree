# Technical documentation

## Searching 

There are two searching model that `sconfig` provide:
- `iter`, will traverse through the whole tree (DFS), this will ensure all the nodes 
  are hit and return a python `generator`.
- `get_contributing_nodes`, similar to `iter` except this will skip the whole sub-graph if 
  sub-graph's `root` is not part of the path, see [Resolving](#resolving) for more details.

Each `iter` function accepts an optional query arguments, if unspecified it'll just 
return all the nodes within this graph.
If specified, this wil determine which nodes match the given query. 

There are two type of queries and one query item used to find the node:
- `Query`, this will compare each query item with the given node
- `ReQuery`, this will do regex match on each query item with the given node
- `QueryItem`, helper class to define the attributes of the node we're searching

```python
node = Node(...)

# get all nodes that has name = "foo"
name_nodes = list(node.iter(query=Query([QueryItem(name="foo")])))
# get all nodes where its name is either "foo", "bar", or "baz"
query = Query([
    QueryItem(name="foo"), 
    QueryItem(name="bar"), 
    QueryItem(name="baz"), 
])
names_nodes = list(node.iter(query=query))

# get all nodes that starts with "foo"
name_start_nodes = list(node.iter(query=ReQuery([QueryItem(name="^foo")])))

# get all nodes that has name = "foo AND weight = 3
name_weight_nodes = list(node.iter(query=Query([QueryItem(name="foo", weight=3)])))

# get an explicit node with path = "/root/tree/foo/bar"
# Note: The following will still requires traversing through the whole node, so 
#       it is recommended to use this if you want to get multiple nodes of 
#       different path. 
path_node = list(node.iter(query=Query([QueryItem(path="/root/tree/foo/bar")])))

# get all nodes that has data items within "foo" node sub-graphs
foo_nodes = list(node.iter(query=Query([QueryItem(name="foo")])))
node_with_data_items = list(
    itertools.chain(
        [node.iter(query=Query([QueryItem(has_data=True)])) for node in foo_nodes]
    )
)
```

## Resolving

A more interesting topic is the main usage of `sconfig`, that is to get all contributing 
nodes from a `path hierarchy`.

> **_NOTE:_** Path hierarchy refers to all the nodes that contributes in the path, this 
  means the set of nodes should be a sub-set of the given path. 
  Syntatically, path hierarchy might looks the same to node path.

For example, say we have the following graph:

```
a
|_b
 |_c
|_b2
  |_c
|_c
```

To get the contributing nodes, we can use the following:

```python
# get the contributing nodes
nodes = node.get_contributing_nodes(path="/a/b/c")
assert nodes == [
    # a/b/c 
    Node(path="a", ...), 
    Node(path="/a/b", parent=Node(path="/a", ...), ...), 
    Node(path="/a/b/c", parent=Node(path="/a/b", ...), ...), 
    # a/c
    Node(path="/a/c", ...)
]
```

As you can see from above, `Node(path="/a/b2", ...)` subgraph is skipped despite having `Node(path="/a/b2/c", ...)`, 
this is because "b2" doesn't contribute in the requested `/a/b/c`. In other words, only the following graph is 
considered from the example above:

```
a
|_b
 |_c
|_c
```

Likewise, if you pass in a path that does not start with the same root, it will return empty

```python
# get the contributing nodes
nodes = node.get_contributing_nodes(path="/b/c")
assert nodes == []  # because it can't find root /b

# get child that 
query = Query([QueryItem(path="/a/b")])
b_node = node.find_child(query)

# now we can use the b child to find the contributing /b and /b/c
b_node.get_contributing_nodes(path="/b/c")
```

The search is a abit naive since we don't want the code to do more than what it should, and the 
above should suffice assuming users would want some search customisation

## Attributes

### Node

- There should never be an identical node amongst its sibling.
  - An identical node is when both `path` and `weight` are equal regardless whether
    it contains any `data item` or not
- Node might have same `name` as long as the final `path` is different.
- `child's weight` is always higher than `parent`.
- Node may/ may not contain `data_item` (including `root` and `leaves`).

### Data Item

## Design decision

There are three data structure candidates to store the node children (see `BaseNode` dataclass).
The first one is to use python `list`, with the advantage of fast mid-index access and secondly 
is to use `deque` (there might be more) with the benefits of O(1) access on either end of the queue.

`sconfig` end up using `deque` because we found that the use case where one need to peek the child 
at certain index should be rare enough.

> Note that I'm not too concerned over the performance as much, if speed really ended up became the main 
  issue over the stability, I'd rather port this lib to its `C++` or `rust` with some python bindings. 

## Implementation details

### Hashing

What are the data we're using to identify if the node is unique?
- `data_item`
  Data item represent unique data contained in the node
- `path`
  Path represent uniqueness of the node, there should never be two nodes that shares the same path
- `weight`
  Weight determines the position among its siblings and the resolving order
- `location`
  Location handles the LazyLoad of the data

What we don't need:
- `name`
- `parent`
- `children`

### Why dataclasses?

Compared to:
- `attrs` lib
- `collections.NamedTuple`
- classic `class` 

### Why descriptors?

The goal is basically to used to ensure the integrity of the `data attributes` from each tree nodes.

For example, lets talk about the `path` attribute. This attribute is used to define the full 
node hierarchy in a static manner using its `name`. 
Technically we could simply use properties like the following:

```python
@dataclass
class Node:
    name: str
    path: str = field(init=False)
    parent: Optional[Node] = None

    @property
    def path(self) -> str:
        return f"{self.parent.path if self.parent else ''}/{self.name}"

    @path.setter
    def path(self, value):
        if value and not isinstance(value, property):
            raise AttributeError("path is read-only.")

n1 = Node("first")
assert n1.path == "/first" 
n2 = Node("second")
assert n2.path == "/second"
n2.parent = n1
assert n2.path == "/first/second"

assert "path" not in n.__dict__
```

Although this works and calling `n.path` returns the expected value, we should also notice that 
path is not part of the class attributes (See last line of code above). While we need both 
`path` and `data_item` value when hashing our node (see [Hasing node](#hashing) for more details). 

Of course there's also another workaround that is to use private variables, e.g: `__path` and have 
the property update that attribute and returns its value; or maybe we could update our hashing 
method to requests the path value. Either way, this would end up creating attribute clutters or 
sporadic logics through the entire code. Lets see the following possible implementation using 
properties.

```python
@dataclass
class Node:
    name: str
    path: str = field(init=False)
    _path: str = field(init=False, repr=False)
    parent: Optional[Node] = None

    @property
    def path(self) -> str:
        return f"{self.parent.path if self.parent else ''}/{self.name}"

    @path.setter
    def path(self, value):
        if value and not isinstance(value, property):
            raise AttributeError("path is read-only.")

assert "_path" in n.__dict__
```

The above will do exactly what we want, that is to always update the path value whenever name changed 
and having this information accessible as class attribute (albeit having to prefix it with underscores)
but it still works, no? Well, here the catch:
- This will still allows any client code to modify the path by accessing `_path` attribute
- The representation of the node will looked different compared to its vars. 

Now, by using descriptor, we're (I think) clearly separating the logic outside and making it easy to 
understand the intention of each attributes without cluttering the dataclass. 
Following is the equivalent of all the above:

```python
class StringDescriptor:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, type):
        """ """
        if obj:
            return obj.__dict__.get(self._name) or ""
        return ""

    def __set__(self, obj, value):
        obj.__dict__[self._name] = value

class PathDescriptor(StringDescriptor):

    @staticmethod
    def __get_path(obj):
        return f"{obj.parent.path if obj.parent else ''}/{obj.name}" if obj else ""

    def __get__(self, obj, objtype=None):
        if obj:
            result = self.__get_path(obj)
            self.__set__(obj, result)
        else:
            result = self._DEFAULT
        return result

    def __set__(self, obj, value):
        obj.__dict__[self._name] = self.__get_path(obj)


@dataclass
class Node:
    name: StringDescriptor = StringDescriptor()
    path: PathDescriptor = PathDescriptor()
    parent: Optional[Node] = None

n = Node("first")
assert n.path == "/first" 
n.name = "second"
assert n.path == "/second"

assert "path" in n.__dict__
```

Although there seemed to have more code, but we should also notice:
- The intention for each attribute is clearer.
- `path` is defined in the instance's dict

See (Descriptor documentation)[https://docs.python.org/3/howto/descriptor.html] for more details.


## Memory management

As most tree-data structure that reference both directions, we all end up with the GC problem.
To avoid memory leak we would want to flag one either the parent/ child as weak reference.

Following are the three possible implementation:
| Weakref | on parent | on child  |
| ------- | --------- | --------- |
| parent  | GC        | Not-GC    | 
| child   | *GC (with exception) | GC | 
| neither | Not-GC    | Not-GC    |

> **GC (with exception)**: 
  when there is a nested tree any node that has children and not bound to any variable will 
  be garbage-collected by the end of the function stack.

  Exhibit A:
  > On each iteration we're dropping any variable that bound any reference to 
    the node that has became a parent of other node, the node will be garbage collected
    and only the last node, Node(D) will be alive because its bound to with ``node`` variable
    and not a parent of any other node.
  ```python
  # Create a node tree of /A/B/C/D
  # A
  # |-B
  #   |-C
  #     |-D
  node = None
  for name in "ABCD":
      n = Node(name)
      if node:
          # In other words, node.add_child(n)
          n.parent = node
      node = n
  ```

  Exhibit B:
  > Deleting the root and only last child is bound to `n` at the end of the iteration 
    all the nodes will also be garbage-collected because they're also a parent of other nodes.
  ```python
  # Create a node tree of /A/B/C/D
  # A
  # |-B
  #   |-C
  #     |-D
  root = Node("A")
  n = root
  for name in "BCD":
      child = Node(name)
      n.add_child(child)
      n = child

  del root
  ```

The behaviour what `sconfig` chose is that `parent` as weakref because its deemed to be 
the safest and behaves the most logical. That is whenever a parent object is deleted all 
unbound child (that is not a parent of other node) will also be garbage-collected.

This also influence the design of the API that is any function that initialized a node 
tree will always return the `parent` node. This will ensure that none of the node is 
deleted. See `create_from_path` for example.
