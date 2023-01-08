from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

from sconfig.config import RENDER_MODE, BaseDataItem, ContextualNode, load_json
from sconfig.query import Query, QueryItem


@dataclass(eq=False)
class CustomDataItem(BaseDataItem):
    """Custom data item that contains additional settings"""

    custom_setting_1: str = ""
    custom_setting_2: int = 0
    custom_setting_3: float = 0.0
    custom_setting_4: bool = False
    custom_setting_5: List[str] = field(default_factory=list)

    def __eq__(self, other):
        return super(CustomDataItem, self).__eq__(other)


@dataclass(eq=False)
class CustomContextualNode(ContextualNode):
    """Custom node that contains custom field"""

    my_additional_attribute: str = ""


def test_custom_node(tmp_path, request):
    """test custom node and items"""

    @dataclass(frozen=True)
    class WeightMaps:
        project: int = 0
        tree: int = 1
        asset: int = 2
        variant: int = 3
        subvariant: int = 4
        lod: int = 5
        type: int = 6
        name: int = 7

        # --- shot context ---
        scene: int = 2
        shot: int = 3
        shotVariant: int = 4
        itemName: int = 5

        # --- additional context ---
        custom_01: int = 6
        custom_02: int = 6
        shot_custom_01: int = 5

    # ACT: create custom contextual node using the weight maps
    context = {
        "project": "projectName",
        "tree": "treeName",
        "scene": "sceneName",
        "shot": "shotName",
        "shotVariant": "shotVariantName",
        "itemName": "itemName",
        "shot_custom_01": "shotCustomName",
    }
    node = CustomContextualNode.create_from_context(
        context, weight_maps=asdict(WeightMaps())
    )

    # ARRANGE: add data item to one of the node
    data_item = CustomDataItem(custom_setting_5=["foo", "bar"])
    shot_custom_01_node = next(
        node.iter(query=Query([QueryItem(name="shotCustomName")]))
    )
    shot_custom_01_node.data_item = data_item

    # ASSERT: constrcuted node has the expected hierarchy and data
    with CustomContextualNode("projectName", 0) as expected_node:
        with expected_node.add_child(CustomContextualNode("treeName", 1)) as tree:
            with tree.add_child(CustomContextualNode("sceneName", 2)) as scene:
                with scene.add_child(CustomContextualNode("shotName", 3)) as shot:
                    with shot.add_child(
                        CustomContextualNode("shotVariantName", 4)
                    ) as variant:
                        with variant.add_child(
                            CustomContextualNode("itemName", 5)
                        ) as item:
                            with item.add_child(
                                CustomContextualNode("shotCustomName", 5)
                            ) as shot_custom:
                                shot_custom.data_item = data_item

    assert list(node.top_node.iter()) == list(expected_node.iter())
