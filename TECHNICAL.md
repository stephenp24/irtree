# Technical documentation

## Design decision

There are three data structure candidates to store the node children (see `BaseNode` dataclass).
The first one is to use python `list`, with the advantage of fast mid-index access and secondly 
is to use `deque` (there might be more) with the benefits of O(1) access on either end of the queue.

`sconfig` end up using `deque` because we're prioritising read access speed over write and the read 
part relies more on the accessibility of either end of the queue.

> Note that I'm not too concerned over the performance as much, if speed really ended up became the main 
  issue over the stability, I'd rather port this lib to its `C++` or `rust` with some python bindings. 

## Implementation details

### Hashing

What are the data we're using to identify if the node is unique?
- `data_item`
- `path`
- `weight`
- `location`

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
    parent: Optional[Node] = None

    @property
    def path(self) -> str:
        return f"{self.parent.path if self.parent else ''}/{self.name}"

n = Node("single")
assert n.path == "/single" 
n.name = "second"
assert n.path == "/second"

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
    _name: str = field(init=False, repr=False)
    path: str = ""
    _path: str = field(init=False, repr=False)
    parent: Optional[Node] = None

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value):
        self._path = f"{self.parent.path if self.parent else ''}/{self.name}"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.path = value

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
class DefaultStringDescriptor:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, type):
        """ """
        if obj:
            return obj.__dict__.get(self._name) or ""
        return ""

    def __set__(self, obj, value):
        obj.__dict__[self._name] = value

class NameDescriptor(DefaultStringDescriptor):

    def __set__(self, obj, value):
      super(NameDescriptor, self).__set__(obj, value)
      obj.path = value

class PathDescriptor(DefaultStringDescriptor):

    def __set__(self, obj, value):
        path = f"{obj.parent.path if obj.parent else ''}/{obj.name}"
        super(NameDescriptor, self).__set__(obj, path)

@dataclass
class Node:
    name: NameDescriptor = NameDescriptor()
    path: PathDescriptor = PathDescriptor()
    parent: Optional[Node] = None

n = Node("first")
assert n.path == "/first" 
n.name = "second"
assert n.path == "/second"
```

Although there seemed to have more code, but the intention for each attributes is clearer.
See (Descriptor documentation)[https://docs.python.org/3/howto/descriptor.html] for more details.
