import os
from typing import Any, Dict, List

import mock
import pytest

from irtree.node import (
    RENDER_MODE,
    BaseDataItem,
    ContextualNode,
    get_resolved_data_item,
    load_json,
)
from irtree.query import Query, QueryItem, ReQuery


@pytest.fixture
def node_with_data_items() -> ContextualNode:
    """fixture that prepare nodes with some data items"""
    nodes_with_data = []  # type: List[ContextualNode]

    a0 = ContextualNode("a0", weight=0)
    nodes_with_data.append(a0)

    b0 = ContextualNode("b0", weight=1)
    c0 = ContextualNode("c0", weight=2)
    a0.add_child(b0)
    b0.add_child(c0)
    nodes_with_data.append(c0)

    b1 = ContextualNode("b1", weight=1)
    c0 = ContextualNode("c0", weight=2)
    d0 = ContextualNode("d0", weight=3)
    a0.add_child(b1)
    b1.add_child(c0)
    b1.add_child(d0)
    nodes_with_data.extend([b1, c0, d0])

    c0 = ContextualNode("c0", weight=2)
    d0 = ContextualNode("d0", weight=3)
    a0.add_child(c0)
    c0.add_child(d0)
    nodes_with_data.extend([c0, d0])

    c1 = ContextualNode("c1", weight=2)
    e2 = ContextualNode("e2", weight=4)
    a0.add_child(c1)
    c1.add_child(e2)
    nodes_with_data.append(e2)

    d0 = ContextualNode("d0", weight=3)
    a0.add_child(d0)
    nodes_with_data.append(d0)

    for node in nodes_with_data:
        item = BaseDataItem(
            author=f"{node.path} (True)",
            total_weight=node.total_weight,
        )
        node.data_item = item

    return a0


def test_context_manager():
    """test constructing nodes as context manager"""
    # ACT: construct node using context manager
    with ContextualNode("a0") as node:
        node.add_child(ContextualNode("b1_0", weight=1))
        with node.add_child(ContextualNode("b1_1", weight=1)) as b1_1:
            with b1_1.add_child(ContextualNode("c2", weight=2)) as c2:
                c2.add_child(ContextualNode("d3_0", weight=3))
                with c2.add_child(ContextualNode("d3_1", weight=3)) as d3_1:
                    with d3_1.add_child(ContextualNode("e4", weight=4)) as e4:
                        with e4.add_child(ContextualNode("f5", weight=5)) as f5:
                            f5.add_child(ContextualNode("g6", weight=6))

    # ASSERT: context manager should construct the same node
    expected_node = ContextualNode("a0")
    expected_node.add_child(ContextualNode("b1_0", weight=1))
    b1 = expected_node.add_child(ContextualNode("b1_1", weight=1))
    c2 = b1.add_child(ContextualNode("c2", weight=2))
    c2.add_child(ContextualNode("d3_0", weight=3))
    d3 = c2.add_child(ContextualNode("d3_1", weight=3))
    e4 = d3.add_child(ContextualNode("e4", weight=4))
    f5 = e4.add_child(ContextualNode("f5", weight=5))
    f5.add_child(ContextualNode("g6", weight=6))

    assert node == expected_node
    assert list(node.iter()) == list(expected_node.iter())


def test_set_parent():
    """test set parent"""
    # ARRANGE: dummy node and parent node
    node = ContextualNode("foo")
    parent_node = ContextualNode("foo")

    # ACT: set parent
    node.parent = parent_node

    # ASSERT: nodes has the the right parent/ child relationship
    assert node != parent_node
    assert node.parent == parent_node
    assert node in parent_node.children


def test_set_parent_error():
    """test set parent to invalid type"""
    # ARRANGE: dummy node
    node = ContextualNode("foo")

    # ASSERT: invalid parent type should raise exception
    with pytest.raises(TypeError):
        node.parent = mock.ANY


def test_add_child():
    """test simple add child"""
    # ARRANGE: dummy node and child
    node = ContextualNode("foo")
    child_node = ContextualNode("foo")

    # ACT: add child
    node.add_child(child_node)

    # ASSERT: nodes has the the right parent/ child relationship
    assert node != child_node
    assert child_node.parent == node
    assert child_node in node.children


def test_add_child_sorted_simple():
    """test add child"""
    # ARRANGE: node with non-identical weight
    node = ContextualNode("root")
    child_node0 = ContextualNode("0", weight=0)
    child_node1 = ContextualNode("1", weight=1)
    child_node2 = ContextualNode("2", weight=2)
    child_node3 = ContextualNode("3", weight=3)
    child_node6 = ContextualNode("6", weight=6)

    # ACT: add child
    node.add_child(child_node1)
    node.add_child(child_node3)
    node.add_child(child_node2)
    node.add_child(child_node6)
    node.add_child(child_node0)

    # ASSERT: children should be sorted based on weight
    expected_result = [
        child_node0,
        child_node1,
        child_node2,
        child_node3,
        child_node6,
    ]
    assert list(node.children) == expected_result

    # ACT: re-add child
    node.add_child(child_node1)
    node.add_child(child_node3)
    node.add_child(child_node2)
    node.add_child(child_node6)

    # ASSERT: no duplicates allowed
    assert list(node.children) == expected_result


def test_add_child_sorted_complex():
    """test add child on identical weight should still sort the child based on
    weight and time of entry properly"""
    # ARRANGE: node with various weights including identical one
    node = ContextualNode("root")
    child_node1 = ContextualNode("1", weight=1)
    child_node2 = ContextualNode("2", weight=2)
    child_node3 = ContextualNode("3", weight=3)
    child_node6 = ContextualNode("6", weight=6)
    child_node2a = ContextualNode("2a", weight=2)
    child_node2b = ContextualNode("2b", weight=2)
    child_node4 = ContextualNode("4", weight=4)
    child_node5 = ContextualNode("5", weight=5)
    child_node5a = ContextualNode("5a", weight=5)
    child_node8 = ContextualNode("8", weight=8)

    # ACT: add the child in unsorted and unordered manner
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

    # ASSERT: add_child should sort based on weight and time of insertion
    expected_result = [
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
    assert list(node.children) == expected_result


def test_add_child_unsorted():
    """test add child unsorted"""
    # ARRANGE: dummy node to add the children
    node = ContextualNode("root")
    children = [ContextualNode("child") for _ in range(3)]

    # ACT: add the children bypassing the sorting
    [node.add_child(child, sort=False) for child in children]

    # ASSERT: duplicates is allowed
    assert len(node.children) == len(children)
    assert list(node.children) == children


def test_add_child_error():
    """test add invalid child"""
    # ARRANGE: dummy node
    node = ContextualNode("foo")

    # ASSERT: invalid child type should raise exception
    with pytest.raises(TypeError):
        node.add_child(mock.ANY)


def test_render(node_with_data_items: ContextualNode):
    """test render both simple and json"""
    # ASSERT: render to term as expected
    term_render = """ |a0 (0) (True)
 |-b0 (1) (False)
  |--c0 (2) (True)
 |-b1 (1) (True)
  |--c0 (2) (True)
  |---d0 (3) (True)
 |--c0 (2) (True)
   |---d0 (3) (True)
 |--c1 (2) (False)
   |----e2 (4) (True)
 |---d0 (3) (True)"""
    assert node_with_data_items.render() == term_render

    # ASSERT: render to json as expected
    json_render = {
        "__kls_name__": "irtree.node.ContextualNode",
        "name": "a0",
        "weight": 0,
        "data_item": {
            "__kls_name__": "irtree.node.BaseDataItem",
            "author": "/a0 (True)",
            "total_weight": 0,
        },
        "children": [
            {
                "__kls_name__": "irtree.node.ContextualNode",
                "name": "b0",
                "weight": 1,
                "data_item": None,
                "children": [
                    {
                        "__kls_name__": "irtree.node.ContextualNode",
                        "name": "c0",
                        "weight": 2,
                        "data_item": {
                            "__kls_name__": "irtree.node.BaseDataItem",
                            "author": "/a0/b0/c0 (True)",
                            "total_weight": 3,
                        },
                        "children": [],
                    }
                ],
            },
            {
                "__kls_name__": "irtree.node.ContextualNode",
                "name": "b1",
                "weight": 1,
                "data_item": {
                    "__kls_name__": "irtree.node.BaseDataItem",
                    "author": "/a0/b1 (True)",
                    "total_weight": 1,
                },
                "children": [
                    {
                        "__kls_name__": "irtree.node.ContextualNode",
                        "name": "c0",
                        "weight": 2,
                        "data_item": {
                            "__kls_name__": "irtree.node.BaseDataItem",
                            "author": "/a0/b1/c0 (True)",
                            "total_weight": 3,
                        },
                        "children": [],
                    },
                    {
                        "__kls_name__": "irtree.node.ContextualNode",
                        "name": "d0",
                        "weight": 3,
                        "data_item": {
                            "__kls_name__": "irtree.node.BaseDataItem",
                            "author": "/a0/b1/d0 (True)",
                            "total_weight": 4,
                        },
                        "children": [],
                    },
                ],
            },
            {
                "__kls_name__": "irtree.node.ContextualNode",
                "name": "c0",
                "weight": 2,
                "data_item": {
                    "__kls_name__": "irtree.node.BaseDataItem",
                    "author": "/a0/c0 (True)",
                    "total_weight": 2,
                },
                "children": [
                    {
                        "__kls_name__": "irtree.node.ContextualNode",
                        "name": "d0",
                        "weight": 3,
                        "data_item": {
                            "__kls_name__": "irtree.node.BaseDataItem",
                            "author": "/a0/c0/d0 (True)",
                            "total_weight": 5,
                        },
                        "children": [],
                    }
                ],
            },
            {
                "__kls_name__": "irtree.node.ContextualNode",
                "name": "c1",
                "weight": 2,
                "data_item": None,
                "children": [
                    {
                        "__kls_name__": "irtree.node.ContextualNode",
                        "name": "e2",
                        "weight": 4,
                        "data_item": {
                            "__kls_name__": "irtree.node.BaseDataItem",
                            "author": "/a0/c1/e2 (True)",
                            "total_weight": 6,
                        },
                        "children": [],
                    }
                ],
            },
            {
                "__kls_name__": "irtree.node.ContextualNode",
                "name": "d0",
                "weight": 3,
                "data_item": {
                    "__kls_name__": "irtree.node.BaseDataItem",
                    "author": "/a0/d0 (True)",
                    "total_weight": 3,
                },
                "children": [],
            },
        ],
    }
    assert node_with_data_items.render(render_mode=RENDER_MODE.JSON) == json_render


def test_get_data_items(node_with_data_items: ContextualNode):
    # ACT: iter data items without any query
    data_items = node_with_data_items.iter_contributing_data_items()

    # ASSERT: iter without query should return all sorted data items from the graph
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

    assert list(data_items) == expected_result

    # ACT: iter data items with continuous query
    data_items = node_with_data_items.iter_contributing_data_items(
        query=Query.from_path("/a0/b1/c0")
    )

    # ASSERT: iter with query should return all sorted items from matching node
    expected_result = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
        BaseDataItem(total_weight=2, author="/a0/c0 (True)"),
        BaseDataItem(total_weight=3, author="/a0/b1/c0 (True)"),
    ]

    assert list(data_items) == expected_result


def test_get_resolved_data_item(node_with_data_items: ContextualNode):
    """test get resolved data_item from contextual node"""
    # ARRANGE: create non-continuous query
    query = Query(
        [
            QueryItem(name="a0", weight=0),
            QueryItem(name="b1", weight=1),
            QueryItem(name="c1", weight=2),
            QueryItem(name="e2", weight=4),
        ]
    )
    # ACT: resolve data item
    data_items = node_with_data_items.iter_contributing_data_items(query=query)
    data_item = get_resolved_data_item(data_items)

    # ASSERT: resulting data_item item is correct
    expected_result = BaseDataItem(total_weight=6, author="/a0/c1/e2 (True)")
    assert data_item == expected_result

    # ARRANGE: create non-continuous query
    query = Query(
        [
            QueryItem(name="a0", weight=0),
            QueryItem(name="b1", weight=1),
            QueryItem(name="d0", weight=3),
        ]
    )
    # ACT: resolve data item
    data_items = node_with_data_items.iter_contributing_data_items(query=query)
    data_item = get_resolved_data_item(data_items)

    # ASSERT: resulting data_item item is correct
    expected_result = BaseDataItem(total_weight=4, author="/a0/b1/d0 (True)")
    assert data_item == expected_result


def test_query(node_with_data_items: ContextualNode):
    """test basic query using query item"""
    # ACT: construct simple query
    query = Query(
        [
            QueryItem(name="a0", weight=0),
            QueryItem(name="b1", weight=1),
            QueryItem(name="c1", weight=2),
            QueryItem(name="e2", weight=4),
        ]
    )
    data_items = node_with_data_items.iter_contributing_data_items(query=query)

    # ASSERT: resulting items is correct
    expected_results = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
        BaseDataItem(total_weight=6, author="/a0/c1/e2 (True)"),
    ]

    assert list(data_items) == expected_results

    # ACT: construct simple query
    query = Query(
        [
            QueryItem(name="a0", weight=0),
            QueryItem(name="b1", weight=1),
            QueryItem(name="d0", weight=3),
        ]
    )
    data_items = node_with_data_items.iter_contributing_data_items(query=query)

    # ASSERT: resulting items is correct
    expected_results = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
        BaseDataItem(total_weight=3, author="/a0/d0 (True)"),
        BaseDataItem(total_weight=4, author="/a0/b1/d0 (True)"),
    ]

    assert list(data_items) == expected_results


def test_query_from_path(node_with_data_items: ContextualNode):
    """test basic query from path constructor"""
    # ACT: initialize query from path
    query = Query.from_path("/a0/b1")
    data_items = node_with_data_items.iter_contributing_data_items(query=query)

    # ASSERT: resulting items is correct
    expected_results = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
    ]

    assert list(data_items) == expected_results


def test_regex_query_from_path(node_with_data_items: ContextualNode):
    """test regex query from path constructor"""
    # ACT: initialize regex query from path
    query = ReQuery.from_path("/a(\\d{1})$/b(\\d{1})$/c0")
    data_items = node_with_data_items.iter_contributing_data_items(query=query)

    # ASSERT: resulting items is correct
    expected_data_items = [
        BaseDataItem(total_weight=0, author="/a0 (True)"),
        BaseDataItem(total_weight=1, author="/a0/b1 (True)"),
        BaseDataItem(total_weight=2, author="/a0/c0 (True)"),
        BaseDataItem(total_weight=3, author="/a0/b0/c0 (True)"),
        BaseDataItem(total_weight=3, author="/a0/b1/c0 (True)"),
    ]

    assert list(data_items) == expected_data_items


def test_top_node():
    """test top_node getter"""
    # ARRANGE: construct simple graph
    with ContextualNode("a0", weight=0) as a0:
        with a0.add_child(ContextualNode("b1", weight=1)) as b1:
            with b1.add_child(ContextualNode("c2", weight=2)) as c2:
                with c2.add_child(ContextualNode("d3", weight=3)) as d3:
                    with d3.add_child(ContextualNode("e4", weight=4)) as e4:
                        with e4.add_child(ContextualNode("f5", weight=5)) as f5:
                            g6 = f5.add_child(ContextualNode("g6", weight=6))

    # ASSERT: all nodes in the graph points to the expected top node
    assert a0.top_node == a0
    assert (
        a0.top_node
        == b1.top_node
        == c2.top_node
        == d3.top_node
        == e4.top_node
        == f5.top_node
        == g6.top_node
        == a0
    )


def test_create_from_path():
    """test node constructor from node path"""
    # ACT: call create from path
    n = ContextualNode.create_from_path("/a0/b1/c2/d3/e4/f5/g6")

    # ASSERT: the create node is as expected
    with ContextualNode("a0", weight=0) as a0:
        with a0.add_child(ContextualNode("b1", weight=1)) as b1:
            with b1.add_child(ContextualNode("c2", weight=2)) as c2:
                with c2.add_child(ContextualNode("d3", weight=3)) as d3:
                    with d3.add_child(ContextualNode("e4", weight=4)) as e4:
                        with e4.add_child(ContextualNode("f5", weight=5)) as f5:
                            g6 = f5.add_child(ContextualNode("g6", weight=6))

    assert n == a0
    assert list(n.iter()) == [a0, b1, c2, d3, e4, f5, g6]


def test_add_data_item():
    """test adding data item"""
    # ARRANGE: dummy node and  data item
    node = ContextualNode("foo")
    item = BaseDataItem("foo")

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
    """test adding invalid data item"""
    # ARRANGE:
    node = ContextualNode("foo", 0)

    # ASSERT: adding invalid data type should raise exception
    with pytest.raises(TypeError):
        node.data_item = mock.ANY


def test_write(node_with_data_items: ContextualNode, tmp_path, request):
    """test writing nodes out as json"""
    # ARRANGE: contruct out file path
    out_dir = tmp_path / request.node.name
    out_dir.mkdir(parents=True)
    out_path = out_dir / "test_write.json"

    # ACT: write nodes out
    node_with_data_items.write(out_path)

    # ASSERT: ensure nodes is written properly
    assert os.path.isfile(out_path)
    assert load_json(out_path) == node_with_data_items


def test_read():
    """test loading saved json node structures"""
    # ACT: load node from saved json file
    node = load_json("./tests/resources/test_read.json")

    # ASSERT: resulting node is as expected
    with ContextualNode("a0") as expected_node:
        with expected_node.add_child(ContextualNode("b1_0", weight=1)) as b1_0:
            with b1_0.add_child(ContextualNode("c2_0", weight=3)) as c2_0:
                c2_0.add_child(ContextualNode("d3_0", weight=3))
                c2_0.add_child(ContextualNode("d3_1", weight=3))
        with expected_node.add_child(ContextualNode("b1_1", weight=1)) as b1_1:
            with b1_1.add_child(ContextualNode("c2", weight=2)) as c2:
                c2.add_child(ContextualNode("d3_0", weight=3))
                with c2.add_child(ContextualNode("d3_1", weight=3)) as d3_1:
                    with d3_1.add_child(ContextualNode("e4", weight=4)) as e4:
                        e4.data_item = BaseDataItem(
                            author="snugroho", total_weight=e4.total_weight
                        )
                        with e4.add_child(ContextualNode("f5", weight=5)) as f5:
                            f5.add_child(ContextualNode("g6", weight=6))
    assert node == expected_node


@pytest.mark.skip("not implemented")
def test_update():
    """ """
