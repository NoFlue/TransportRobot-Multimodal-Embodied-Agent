"""관측(ObservationDTO)을 에이전트가 읽을 구조화된 텍스트로 바꾼다."""

from __future__ import annotations

from agent.dto import ObservationDTO


def object_kind(label: str) -> str:
    """label 접두어가 종류다. 예: coal_1 -> coal"""
    stem, _, suffix = label.rpartition("_")
    return stem if stem and suffix.isdigit() else label


def describe_world(observation: ObservationDTO, near_radius: float) -> str:
    agent = observation.agent
    carried = {agent.holding, *agent.inventory} - {None}
    carried_text = [f"{agent.holding} (in hand)"] if agent.holding else []
    carried_text += [f"{label} (in inventory)" for label in agent.inventory]
    lines = [
        "[agent]",
        f"position={agent.position}",
        f"you carry {len(carried_text)} object(s): {', '.join(carried_text) or 'nothing'}",
        "[objects] movable = can be picked up, fixed = cannot; distance = horizontal meters from you",
    ]
    for obj in observation.objects:
        kind = "movable" if obj.pickable else "fixed"
        if obj.label == agent.holding:
            status = "in your hand"
        elif obj.label in agent.inventory:
            status = "in your inventory"
        else:
            status = f"on the ground, distance={obj.position.horizontal_distance(agent.position):.1f}"
        lines.append(f"- {obj.label} type={object_kind(obj.label)} {kind} position={obj.position} {status}")

    lines.append(f"[relations] for each object on the ground: other objects within {near_radius:.1f} m (horizontal, between centers)")
    on_ground = [o for o in observation.objects if o.label not in carried]
    for obj in on_ground:
        others = sorted(
            (obj.position.horizontal_distance(other.position), other.label) for other in on_ground if other is not obj
        )
        near = [f"near {label} ({distance:.1f} m)" for distance, label in others if distance <= near_radius]
        if near:
            lines.append(f"- {obj.label}: {', '.join(near)}")
        elif others:
            distance, label = others[0]
            lines.append(f"- {obj.label}: nothing near (closest {label} {distance:.1f} m)")
    return "\n".join(lines)
