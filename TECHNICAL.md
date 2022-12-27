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
