from collections import deque
from typing import Any, Dict, List

import mock
import pytest

from sconfig._profiler import profiler
from sconfig.config import BaseDataItem
from sconfig.config import ConfigNode as _N
from sconfig.config import walk, get_resolved_data_item
from sconfig.query import QueryItem, Query, ReQuery, get_query_items_from_path


@pytest.fixture(autouse=True)
def profile(request: pytest.FixtureRequest):
    """Pytest fixture to capture the profile stats at the end of the test"""
    yield
    # if request.config.getoption("verbose") > 0:
    #     profiler.print_stats()


@pytest.fixture
def sample_node0() -> _N:
    """Pytest fixture that prepare the test nodes"""
    nodes_with_data = []  # type: List[_N]

    a0 = _N("a0", weight=0)
    nodes_with_data.append(a0)

    b0 = _N("b0", weight=1)
    c0 = _N("c0", weight=2)
    a0.add_child(b0)
    b0.add_child(c0)
    nodes_with_data.append(c0)

    b1 = _N("b1", weight=1)
    c0 = _N("c0", weight=2)
    d0 = _N("d0", weight=3)
    a0.add_child(b1)
    b1.add_child(c0)
    b1.add_child(d0)
    nodes_with_data.extend([b1, c0, d0])

    c0 = _N("c0", weight=2)
    d0 = _N("d0", weight=3)
    a0.add_child(c0)
    c0.add_child(d0)
    nodes_with_data.extend([c0, d0])

    c1 = _N("c1", weight=2)
    e2 = _N("e2", weight=4)
    a0.add_child(c1)
    c1.add_child(e2)
    nodes_with_data.append(e2)

    d0 = _N("d0", weight=3)
    a0.add_child(d0)
    nodes_with_data.append(d0)

    for node in nodes_with_data:
        item = BaseDataItem(
            author=f"{node.path} (True)",
            total_weight=node.total_weight,
        )
        node.data_item = item

    return a0


def test_get_config(sample_node0: _N):
    data_items = sample_node0.get_configs(query=Query(get_query_items_from_path("/a0/b1/c0")))
    for data_item in data_items:
        print(data_item)
    assert True


def test_set_parent():
    node = _N("foo")
    parent_node = _N("foo")
    node.parent = parent_node

    # ASSERT: root and child has the the right parent/ child relationship
    assert node != parent_node
    assert node.parent == parent_node
    assert node in parent_node.children


def test_set_parent_error():
    node = _N("foo")

    # ASSERT: invalid parent type should raise exception
    with pytest.raises(TypeError):
        node.parent = mock.ANY


def test_add_child():
    node = _N("foo")
    child_node = _N("foo")
    node.add_child(child_node)

    # ASSERT: root and child has the the right parent/ child relationship
    assert node != child_node
    assert child_node.parent == node
    assert child_node in node.children


def test_add_child_sorted_simple():
    node = _N("root")
    child_node1 = _N("1", weight=1)
    child_node2 = _N("2", weight=2)
    child_node3 = _N("3", weight=3)
    child_node6 = _N("6", weight=6)
    node.add_child(child_node1)
    node.add_child(child_node3)
    node.add_child(child_node2)
    node.add_child(child_node6)

    child_node0 = _N("0", weight=0)
    node.add_child(child_node0)

    expected_result = deque(
        [
            child_node0,
            child_node1,
            child_node2,
            child_node3,
            child_node6,
        ]
    )

    # ASSERT: children should be sorted based on weight
    node.render()
    assert node.children == expected_result

    node.add_child(child_node1)
    node.add_child(child_node3)
    node.add_child(child_node2)
    node.add_child(child_node6)

    # ASSERT: children should avoid duplicates
    node.render()
    assert node.children == expected_result


def test_add_child_sorted_complex():
    node = _N("root")
    child_node1 = _N("1", weight=1)
    child_node2 = _N("2", weight=2)
    child_node3 = _N("3", weight=3)
    child_node6 = _N("6", weight=6)
    child_node2a = _N("2a", weight=2)
    child_node2b = _N("2b", weight=2)
    child_node4 = _N("4", weight=4)
    child_node5 = _N("5", weight=5)
    child_node5a = _N("5a", weight=5)
    child_node8 = _N("8", weight=8)
    node.add_child(child_node1)
    node.add_child(child_node3)
    node.add_child(child_node2)
    node.add_child(child_node6)
    node.add_child(child_node2b)
    node.add_child(child_node2a)
    node.add_child(child_node5)
    node.add_child(child_node5a)
    node.add_child(child_node4)
    node.add_child(child_node8)

    expected_result = deque(
        [
            child_node1,
            child_node2,
            child_node2b,
            child_node2a,
            child_node3,
            child_node4,
            child_node5,
            child_node5a,
            child_node6,
            child_node8,
        ]
    )

    # ASSERT: same weight should be sorted and inserted after the first occurences
    node.render()
    assert node.children == expected_result


def test_add_child_unsafe():
    node = _N("root")
    children = [_N("child") for _ in range(3)]
    [node.add_child(child, safe=False) for child in children]

    expected_result = deque(children)

    # ASSERT: same weight should be sorted and inserted after the first occurences
    node.render()
    assert node.children == expected_result


def test_add_child_error():
    node = _N("foo")

    # ASSERT: invalid child type should raise exception
    with pytest.raises(TypeError):
        node.add_child(mock.ANY)


def test_render(sample_node0: _N):
    sample_node0.render()

    assert True


def test_get_configs(sample_node0: _N):
    configs = sample_node0.get_configs()
    expected_result = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
        BaseDataItem(total_weight=2, author="/a0/c0 (True)"),
        BaseDataItem(total_weight=3, author="/a0/b0/c0 (True)"),
        BaseDataItem(total_weight=3, author="/a0/b1/c0 (True)"),
        BaseDataItem(total_weight=3, author="/a0/d0 (True)"),
        BaseDataItem(total_weight=4, author="/a0/b1/d0 (True)"),
        BaseDataItem(total_weight=5, author="/a0/c0/d0 (True)"),
        BaseDataItem(total_weight=6, author="/a0/c1/e2 (True)"),
    ]
    print(sample_node0.render())

    assert list(configs) == expected_result


def test_get_resolved_config(sample_node0: _N):
    query = Query(
        [
            QueryItem(name="a0", weight=0),
            QueryItem(name="b1", weight=1),
            QueryItem(name="c1", weight=2),
            QueryItem(name="e2", weight=4),
        ]
    )
    configs = sample_node0.get_configs(query=query)
    config = get_resolved_data_item(configs)
    expected_result = BaseDataItem(total_weight=6, author="/a0/c1/e2 (True)")
    # ASSERT: resulting config item is correct
    assert config == expected_result

    query = Query(
        [
            QueryItem(name="a0", weight=0),
            QueryItem(name="b1", weight=1),
            QueryItem(name="d0", weight=3),
        ]
    )
    sample_node0.render()
    configs = sample_node0.get_configs(query=query)
    config = get_resolved_data_item(configs)
    expected_result = BaseDataItem(total_weight=4, author="/a0/b1/d0 (True)")
    # ASSERT: resulting config item is correct
    assert config == expected_result


def test_exact_query(sample_node0: _N):
    query = Query(
        [
            QueryItem(name="a0", weight=0),
            QueryItem(name="b1", weight=1),
            QueryItem(name="c1", weight=2),
            QueryItem(name="e2", weight=4),
        ]
    )
    configs = sample_node0.get_configs(query=query)
    expected_results = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
        BaseDataItem(total_weight=6, author="/a0/c1/e2 (True)"),
    ]

    # ASSERT: resulting config items is correct
    assert list(configs) == expected_results

    query = Query(
        [
            QueryItem(name="a0", weight=0),
            QueryItem(name="b1", weight=1),
            QueryItem(name="d0", weight=3),
        ]
    )
    configs = sample_node0.get_configs(query=query)
    expected_results = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
        BaseDataItem(total_weight=3, author="/a0/d0 (True)"),
        BaseDataItem(total_weight=4, author="/a0/b1/d0 (True)"),
    ]

    # ASSERT: resulting config items is correct
    assert list(configs) == expected_results


def test_exact_query_from_path(sample_node0: _N):
    query = Query(get_query_items_from_path("/a0/b1"))
    configs = sample_node0.get_configs(query=query)

    expected_results = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
    ]

    # ASSERT: resulting config items is correct
    assert list(configs) == expected_results


def test_regex_query_from_path(sample_node0: _N):
    query = ReQuery(get_query_items_from_path("/a(\\d{1})$/b(\\d{1})$/c0"))
    configs = sample_node0.get_configs(query=query)

    expected_configs = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
        BaseDataItem(total_weight=2, author="/a0/c0 (True)"),
        BaseDataItem(total_weight=3, author="/a0/b0/c0 (True)"),
        BaseDataItem(total_weight=3, author="/a0/b1/c0 (True)"),
    ]

    # ASSERT: resulting config items is correct
    assert list(configs) == expected_configs


def test_root():
    a0 = _N("a0", weight=0)
    b1 = _N("b1", weight=1)
    c2 = _N("c2", weight=2)
    d3 = _N("d3", weight=3)
    e4 = _N("e4", weight=4)
    f5 = _N("f5", weight=5)
    g6 = _N("g6", weight=6)
    a0.add_child(b1)
    b1.add_child(c2)
    c2.add_child(d3)
    d3.add_child(e4)
    e4.add_child(f5)
    f5.add_child(g6)
    assert a0.root == a0
    assert (
        a0.root
        == b1.root
        == c2.root
        == d3.root
        == e4.root
        == f5.root
        == g6.root
        == a0
    )


def test_create_from_path():
    n = _N.create_from_path("/a0/b1/c2/d3/e4/f5/g6")

    a0 = _N("a0", weight=0)
    b1 = _N("b1", weight=1)
    c2 = _N("c2", weight=2)
    d3 = _N("d3", weight=3)
    e4 = _N("e4", weight=4)
    f5 = _N("f5", weight=5)
    g6 = _N("g6", weight=6)
    a0.add_child(b1)
    b1.add_child(c2)
    c2.add_child(d3)
    d3.add_child(e4)
    e4.add_child(f5)
    f5.add_child(g6)

    a0.render()

    assert n == a0
    assert list(n.iter(Query([QueryItem.from_node(g6)]))) == [g6]


@pytest.mark.parametrize(
    "path,item", 
    [
        ("/a0/b1", BaseDataItem("b1", 1)), 
        ("/a0/b1/c2/d3", BaseDataItem("d3", 3)), 
        ("/a0/b1/c2/d3/e4/f5/g6", BaseDataItem("g6", 6)),
    ], 
)
def test_add_data_item(path, item):
    """ """
    root_node = _N.create_from_path(path)
    query = Query(get_query_items_from_path(path)[-1:])
    node = next(root_node.iter(query))

    # ASSERT: node is empty
    assert node.has_data is False
    
    # ACT: add data item
    node.data_item = item

    # ASSERT: node has data item as expected
    assert node.has_data is True
    assert node.data_item == item

    # ACT: emptying data item
    node.data_item = None

    # ASSERT: child node is empty
    assert node.has_data is False
    assert node.data_item is None


def test_add_invalid_data_item():
    node = _N("foo", 0)
    with pytest.raises(TypeError):
        node.data_item = mock.ANY


from dataclasses import dataclass, field

@dataclass
class TestItem(BaseDataItem):
    key_0: str = ""
    key_1: bool = False
    key_2: int = 0
    key_3: float = 0.0
    key_4: list = field(default_factory=list)
    key_5: set = field(default_factory=set)
    key_6: Dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def complex_node() -> List[_N]:
    paths = ("/a0/b1_0", "/a0/b1_1/c2/d3_0", "/a0/b1_1/c2/d3_1/e4/f5/g6")
    result = [_N.create_from_path(path) for path in paths]
    for path in paths:
        node = _N.create_from_path(path)
        query = Query(get_query_items_from_path(path)[-1:])
        result += list(node.iter(query))

    return result


def _test_union(complex_node):
    n_0, n_1, n_2 = complex_node
    print(f"{n_0!r}\n{n_1!r}\n{n_2!r}")

    item_0 = TestItem(
        "item_0",
        True,
        190,
        12.0,
    )

    path = "/a0/b1"
    assert n_0.has_data is False
    n_0.data_item = item_0

    assert n_0.has_data is True
    assert n_0.data_item == item_0

    path = "/a0/b1/c2/d3"
    item_1 = TestItem("item_1", key_5={"item_1", 191, 12.1})

    assert n_1.has_data is False
    assert n_1.path == path 

    n_1.data_item = item_1

    assert n_1.has_data is True
    assert n_1.data_item == item_1

    path = "/a0/b1/c2/d3/e4/f5/g6"
    item_2 = TestItem(
        key_4=["item_2", 191, 12.1],
        key_5={19, 12, 2019},
        key_6={"date": "19-12-2019"},
    )

    assert n_2.has_data is False
    assert n_2.path == path

    n_2.data_item = item_2

    assert n_2.has_data is True
    assert n_2.data_item == item_2

