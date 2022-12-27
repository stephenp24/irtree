from collections import deque
from typing import Any, Dict

import mock
import pytest

from sconfig._profiler import profiler
from sconfig.config import BaseDataItem
from sconfig.config import ConfigNode as _N
from sconfig.config import walk
from sconfig.query import Component, ExactQuery, RegexQuery


@pytest.fixture(autouse=True)
def profile(request: pytest.FixtureRequest):
    """Pytest fixture to capture the profile stats at the end of the test"""
    yield
    # if request.config.getoption("verbose") > 0:
    #     profiler.print_stats()


@pytest.fixture
def sample_node0() -> _N:
    """Pytest fixture that prepare the test nodes"""
    a0 = _N("a0", weight=0, has_data=True)

    b0 = _N("b0", weight=1)
    c0 = _N("c0", weight=2, has_data=True)
    a0.add_child(b0)
    b0.add_child(c0)

    b1 = _N("b1", weight=1, has_data=True)
    c0 = _N("c0", weight=2, has_data=True)
    d0 = _N("d0", weight=3, has_data=True)
    a0.add_child(b1)
    b1.add_child(c0)
    b1.add_child(d0)

    c0 = _N("c0", weight=2, has_data=True)
    d0 = _N("d0", weight=3, has_data=True)
    a0.add_child(c0)
    c0.add_child(d0)

    c1 = _N("c1", weight=2)
    e2 = _N("e2", weight=4, has_data=True)
    a0.add_child(c1)
    c1.add_child(e2)

    d0 = _N("d0", weight=3, has_data=True)
    a0.add_child(d0)

    for node in walk(a0):
        if node.has_data:
            prefix = f"{node.get_parent()!s}/" if node.get_parent() else ""
            item = BaseDataItem(
                author=f"{prefix}{node!s}",
                sum_weight=node.get_sum_weight(),
            )
            node.add_data_item(item)

    return a0


def test_set_parent():
    node = _N("foo")
    parent_node = _N("foo")
    node.set_parent(parent_node)

    # ASSERT: root and child has the the right parent/ child relationship
    assert node != parent_node
    assert node.get_parent() == parent_node
    assert node in parent_node.get_children()


def test_set_parent_error():
    node = _N("foo")

    # ASSERT: invalid parent type should raise exception
    with pytest.raises(TypeError):
        node.set_parent(mock.ANY)


def test_add_child():
    node = _N("foo")
    child_node = _N("foo")
    node.add_child(child_node)

    # ASSERT: root and child has the the right parent/ child relationship
    assert node != child_node
    assert child_node.get_parent() == node
    assert child_node in node.get_children()


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
    node.as_graph()
    assert node.get_children() == expected_result

    node.add_child(child_node1)
    node.add_child(child_node3)
    node.add_child(child_node2)
    node.add_child(child_node6)

    # ASSERT: children should avoid duplicates
    node.as_graph()
    assert node.get_children() == expected_result


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
    node.as_graph()
    assert node.get_children() == expected_result


def test_add_child_unsafe():
    node = _N("root")
    children = [_N("child") for _ in range(3)]
    [node.add_child(child, safe=False) for child in children]

    expected_result = deque(children)

    # ASSERT: same weight should be sorted and inserted after the first occurences
    node.as_graph()
    assert node.get_children() == expected_result


def test_add_child_error():
    node = _N("foo")

    # ASSERT: invalid child type should raise exception
    with pytest.raises(TypeError):
        node.add_child(mock.ANY)


def test_as_graph(sample_node0: _N):
    sample_node0.as_graph()

    assert True


def test_get_configs(sample_node0: _N):
    configs = sample_node0.get_configs()
    expected_result = [
        BaseDataItem(sum_weight=0, author="a0 (0) (True)"),
        BaseDataItem(sum_weight=1, author="a0 (0) (True)/b1 (1) (True)"),
        BaseDataItem(sum_weight=2, author="a0 (0) (True)/c0 (2) (True)"),
        BaseDataItem(sum_weight=3, author="b0 (1) (False)/c0 (2) (True)"),
        BaseDataItem(sum_weight=3, author="b1 (1) (True)/c0 (2) (True)"),
        BaseDataItem(sum_weight=3, author="a0 (0) (True)/d0 (3) (True)"),
        BaseDataItem(sum_weight=4, author="b1 (1) (True)/d0 (3) (True)"),
        BaseDataItem(sum_weight=5, author="c0 (2) (True)/d0 (3) (True)"),
        BaseDataItem(sum_weight=6, author="c1 (2) (False)/e2 (4) (True)"),
    ]
    print(sample_node0.as_graph())

    assert list(configs) == expected_result


def test_get_resolved_config(sample_node0: _N):
    query = ExactQuery(
        [
            Component(0, "a0"),
            Component(1, "b1"),
            Component(2, "c1"),
            Component(4, "e2"),
        ]
    )
    config = sample_node0.get_resolved_config(query=query)
    expected_result = BaseDataItem(sum_weight=6, author="c1 (2) (False)/e2 (4) (True)")
    # ASSERT: resulting config item is correct
    assert config == expected_result

    query = ExactQuery(
        [
            Component(0, "a0"),
            Component(1, "b1"),
            Component(3, "d0"),
        ]
    )
    config = sample_node0.get_resolved_config(query=query)
    expected_result = BaseDataItem(sum_weight=4, author="b1 (1) (True)/d0 (3) (True)")
    # ASSERT: resulting config item is correct
    assert config == expected_result


def test_exact_query(sample_node0: _N):
    query = ExactQuery(
        [
            Component(0, "a0"),
            Component(1, "b1"),
            Component(2, "c1"),
            Component(4, "e2"),
        ]
    )
    configs = sample_node0.get_configs(query=query)
    expected_results = [
        BaseDataItem(sum_weight=0, author="a0 (0) (True)"),
        BaseDataItem(sum_weight=1, author="a0 (0) (True)/b1 (1) (True)"),
        BaseDataItem(sum_weight=6, author="c1 (2) (False)/e2 (4) (True)"),
    ]

    # ASSERT: resulting config items is correct
    assert list(configs) == expected_results

    query = ExactQuery(
        [
            Component(0, "a0"),
            Component(1, "b1"),
            Component(3, "d0"),
        ]
    )
    configs = sample_node0.get_configs(query=query)
    expected_results = [
        BaseDataItem(sum_weight=0, author="a0 (0) (True)"),
        BaseDataItem(sum_weight=1, author="a0 (0) (True)/b1 (1) (True)"),
        BaseDataItem(sum_weight=3, author="a0 (0) (True)/d0 (3) (True)"),
        BaseDataItem(sum_weight=4, author="b1 (1) (True)/d0 (3) (True)"),
    ]

    # ASSERT: resulting config items is correct
    assert list(configs) == expected_results


def test_exact_query_from_node_path(sample_node0: _N):
    query = ExactQuery.from_node_path("/a0/b1")
    configs = sample_node0.get_configs(query=query)

    expected_results = [
        BaseDataItem(sum_weight=0, author="a0 (0) (True)"),
        BaseDataItem(sum_weight=1, author="a0 (0) (True)/b1 (1) (True)"),
    ]

    # ASSERT: resulting config items is correct
    assert list(configs) == expected_results


def test_regex_query_from_node_path(sample_node0: _N):
    query = RegexQuery.from_node_path("/a(\\d{1})$/b(\\d{1})$/c0")
    configs = sample_node0.get_configs(query=query)

    expected_configs = [
        BaseDataItem(sum_weight=0, author="a0 (0) (True)"),
        BaseDataItem(sum_weight=1, author="a0 (0) (True)/b1 (1) (True)"),
        BaseDataItem(sum_weight=2, author="a0 (0) (True)/c0 (2) (True)"),
        BaseDataItem(sum_weight=3, author="b0 (1) (False)/c0 (2) (True)"),
        BaseDataItem(sum_weight=3, author="b1 (1) (True)/c0 (2) (True)"),
    ]

    # ASSERT: resulting config items is correct
    assert list(configs) == expected_configs


def test_root():
    # expected_root = _N("a0", weight=0, has_data=True)
    # expected_root.add_data_item(BaseDataItem(
    #     author=f"{expected_root!s}",
    #     sum_weight=expected_root.get_sum_weight(),
    # ))
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
    assert a0.get_root() == a0
    assert (
        a0.get_root()
        == b1.get_root()
        == c2.get_root()
        == d3.get_root()
        == e4.get_root()
        == f5.get_root()
        == g6.get_root()
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

    n.as_graph()

    a0.as_graph()

    assert n == g6


def test_add_data_item():

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

    item_0 = TestItem(
        "item_0",
        True,
        190,
        12.0,
    )
    n_0 = _N.create_from_path("/a0/b1")
    n_0.add_data_item(item_0)

    assert n_0.get_data_item() == item_0

    item_1 = TestItem("item_1", key_5={"item_1", 191, 12.1})
    n_1 = _N.create_from_path("/a0/b1/c2/d3")
    n_1.add_data_item(item_1)

    assert n_1.get_data_item() == item_1

    item_2 = TestItem(
        key_4=["item_2", 191, 12.1],
        key_5={19, 12, 2019},
        key_6={"date": "19-12-2019"},
    )
    n_2 = _N.create_from_path("/a0/b1/c2/d3/e4/f5/g6")
    n_2.add_data_item(item_2)

    assert n_2.get_data_item() == item_2
