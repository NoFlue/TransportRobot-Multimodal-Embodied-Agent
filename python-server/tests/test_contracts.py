import json
from pathlib import Path

import pytest

from agent.dto import ActionResultDTO, ActRequestDTO, ObservationDTO
from agent.tools import ToolCallError, to_act_request
from agent.world_text import describe_world, object_kind

CONTRACTS = Path(__file__).resolve().parents[2] / "docs" / "contracts"


def load(name: str) -> dict:
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def test_contract_examples_parse_into_dtos():
    observation = ObservationDTO.model_validate(load("observation.example.json"))
    request = ActRequestDTO.model_validate(load("act_request.example.json"))
    result = ActionResultDTO.model_validate(load("action_result.example.json"))

    assert observation.find("furnace_1") is not None
    assert observation.message.startswith("석탄")
    assert request.model_dump(exclude_none=True) == load("act_request.example.json")
    assert result.observation.agent.position.z == pytest.approx(12.9)


def test_world_text_lists_objects_and_relations():
    observation = ObservationDTO.model_validate(load("observation.example.json"))
    observation.agent.holding = "coal_1"
    observation.objects[3].position.x = 1.5  # iron_1 next to furnace_1
    observation.objects[3].position.z = 5.0

    text = describe_world(observation, near_radius=3.0)

    assert "you carry 1 object(s): coal_1 (in hand)" in text
    assert "- coal_1 type=coal movable" in text and "- furnace_1 type=furnace fixed" in text
    assert "- iron_1: near furnace_1 (1.8 m)" in text
    assert "- furnace_1: near iron_1 (1.8 m)" in text
    relations = text.split("[relations]", 1)[1]
    assert "coal_1" not in relations  # 든 물건은 관계에서 빠진다
    assert "- coal_2: nothing near (closest " in text


def test_object_kind_uses_label_prefix():
    assert object_kind("coal_12") == "coal"
    assert object_kind("furnace") == "furnace"


def test_tool_call_validation():
    assert to_act_request("drop", {"label": "coal_1", "near_label": "furnace_1"}).near_label == "furnace_1"
    assert to_act_request("store", {}).label is None
    with pytest.raises(ToolCallError):
        to_act_request("fly", {})
    with pytest.raises(ToolCallError):
        to_act_request("pick", {})
