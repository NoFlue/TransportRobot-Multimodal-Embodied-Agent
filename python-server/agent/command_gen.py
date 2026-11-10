"""실험 명령 생성기: 시드와 현재 관측으로 명령, 필수 단계, 성공 조건, 준비 동작을 만든다.

영역(single·double·complex·abstract) 정의에 맞는 명령을 무작위로 만들고, 대상은 정확한 label, 종류("a piece of coal"),
거리("the nearest iron ore") 중 하나로 부른다. 같은 시드와 같은 배치면 같은 명령이 나온다.
평가 전용이다. 에이전트 프롬프트·검사 규칙을 이 명령에 맞춰 고치지 않는다.

반환값은 experiment.TaskSpec과 같은 모양의 dict다. reference는 정답 행동을 실제 label로 적은 것(완벽한 에이전트의 실행)이다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from agent.dto import ObservationDTO
from agent.world_text import object_kind

ANY = "any"
# 종류 -> (단수 명사, 부정관사 표현들)
NOUNS = {
    "coal": ("coal", ["a coal", "a piece of coal", "a lump of coal"]),
    "iron": ("iron ore", ["an iron ore", "a piece of iron ore", "a chunk of iron ore"]),
    "furnace": ("furnace", ["the furnace"]),
}

MOVE = ["Go to {X}.", "Walk over to {X}.", "Head to {X}.", "Move next to {X}.", "Approach {X}.",
        "Make your way to {X}.", "Walk up to {X}.", "Go stand near {X}.", "Navigate to {X}.", "Get close to {X}."]
LOOK = ["Look at {X}.", "Face {X}.", "Turn toward {X}.", "Turn to face {X}.", "Glance at {X}.", "Look toward {X}.",
        "Rotate to look at {X}.", "Point your camera at {X}.", "Turn your head toward {X}.", "Direct your gaze at {X}."]
PICK = ["Pick up {X}.", "Grab {X}.", "Take {X}.", "Collect {X}.", "Lift {X} off the ground.", "Get {X}.",
        "Fetch {X}.", "Pick {X} up."]
DROP = ["Put {X} down.", "Drop {X}.", "Set {X} down on the ground.", "Place {X} on the ground.",
        "Leave {X} on the ground.", "Release {X}."]
DROP_NEAR = ["Put {X} next to {Y}.", "Drop {X} beside {Y}.", "Place {X} near {Y}.", "Leave {X} by {Y}.",
             "Set {X} down next to {Y}."]
STORE = ["Store {X} in your inventory.", "Put {X} in your inventory.", "Stash {X} in your inventory.",
         "Put {X} away in your inventory.", "Move {X} from your hand into your inventory.", "Keep {X} in your inventory."]
CARRY = ["Bring {X} to {Y}.", "Carry {X} over to {Y}.", "Take {X} to {Y}.", "Move {X} next to {Y}.", "Deliver {X} to {Y}."]
JOIN = [", then ", " and then ", ". Then ", ", and after that ", ". After that, "]
HELD = ["the {noun} you are holding", "the {noun} you're carrying", "the {noun} in your hand"]
TEMPLATES = {"move": MOVE, "look": LOOK, "pick": PICK, "drop": DROP, "drop_near": DROP_NEAR, "store": STORE, "carry": CARRY}


@dataclass
class Draft:
    """준비 동작과, 준비가 끝난 관측으로 명령을 완성하는 함수."""

    setup: list[dict]
    finalize: Callable[[ObservationDTO], dict]


class World:
    def __init__(self, observation: ObservationDTO):
        self.agent = observation.agent.position
        self.objects = {o.label: o for o in observation.objects if object_kind(o.label) in NOUNS}
        self.carried = [label for label in [observation.agent.holding, *observation.agent.inventory] if label in self.objects]
        self.holding = observation.agent.holding if observation.agent.holding in self.objects else None
        self.inventory = [label for label in observation.agent.inventory if label in self.objects]

    def pickable(self) -> list[str]:
        return sorted(label for label, o in self.objects.items() if o.pickable)

    def free(self) -> list[str]:
        return sorted(label for label in self.objects if label not in self.carried)

    def of_kind(self, kind: str) -> list[str]:
        return [label for label in self.free() if object_kind(label) == kind]

    def distance(self, label: str) -> float:
        return self.agent.horizontal_distance(self.objects[label].position)

    def nearest(self, labels: list[str]) -> str:
        return min(labels, key=self.distance)


def _sentence(text: str) -> str:
    return text[0].upper() + text[1:]


def _clause(template: str) -> str:
    return template[0].lower() + template[1:].rstrip(".")


def _step(tool: str, label: str | None = None, near: str | None = None) -> dict:
    step = {"tool": tool}
    if label is not None:
        step["label"] = label
    if near is not None:
        step["near_label"] = near
    return step


def _act(action: str, label: str | None = None, near: str | None = None) -> dict:
    return {"action": action, "label": label, "near_label": near}


# ---------------- 행동 절 조합 (single·double·complex) ----------------


@dataclass
class Clause:
    action: str  # move, look, pick, store, drop, drop_near, carry
    obj: str | None = None
    anchor: str | None = None

    @property
    def steps(self) -> int:
        return 2 if self.action == "carry" else 1


@dataclass
class State:
    holding: str | None
    inventory: list[str]
    anchors: set[str] = field(default_factory=set)  # 내려놓을 자리로 쓴 객체. 나중에 옮기지 않는다.
    placed: set[str] = field(default_factory=set)  # 이 명령에서 내려놓은 객체. 다시 줍지 않는다.
    location: str | None = None  # 마지막으로 걸어간 대상

    @property
    def carried(self) -> list[str]:
        return ([self.holding] if self.holding else []) + self.inventory


def _candidates(world: World, state: State, remaining: int, previous: Clause | None) -> dict[str, list[Clause]]:
    carried = state.carried
    free = [label for label in world.objects if label not in carried]
    movable = [label for label in world.pickable() if label not in carried and label not in state.placed | state.anchors]
    # 방금 인벤토리에 넣은 물체를 바로 꺼내 내려놓는 어색한 명령은 만들지 않는다.
    just_stored = previous.obj if previous and previous.action == "store" else None
    droppable = [x for x in carried if x != just_stored]
    options = {
        "move": [Clause("move", x) for x in free if x != state.location],
        "look": [Clause("look", x) for x in free if not (previous and previous.action == "look" and previous.obj == x)],
        "pick": [Clause("pick", x) for x in movable if len(carried) < 5],
        "store": [Clause("store", state.holding)] if state.holding else [],
        "drop": [Clause("drop", x) for x in droppable],
        "drop_near": [Clause("drop_near", x, y) for x in droppable for y in free],
        "carry": [Clause("carry", x, y) for x in movable for y in free if y != x] if remaining >= 2 and len(carried) < 5 else [],
    }
    return {name: items for name, items in options.items() if items}


def _apply(state: State, clause: Clause) -> None:
    if clause.action == "move":
        state.location = clause.obj
    elif clause.action in ("pick", "carry"):
        if state.holding:
            state.inventory.append(state.holding)
        state.holding = clause.obj
        state.location = clause.obj
        if clause.action == "carry":
            _drop(state, clause.obj)
            state.placed.add(clause.obj)
            state.anchors.add(clause.anchor)
            state.location = clause.anchor
    elif clause.action == "store":
        state.inventory.append(state.holding)
        state.holding = None
    elif clause.action in ("drop", "drop_near"):
        _drop(state, clause.obj)
        state.placed.add(clause.obj)
        if clause.action == "drop_near":
            state.anchors.add(clause.anchor)
            state.location = clause.anchor


def _drop(state: State, label: str) -> None:
    if state.holding == label:
        state.holding = None
    else:
        state.inventory.remove(label)


def _compose(rng: random.Random, world: World, total: int) -> list[Clause]:
    for _ in range(200):
        state = State(world.holding, list(world.inventory))
        clauses: list[Clause] = []
        steps = 0
        while steps < total:
            options = _candidates(world, state, total - steps, clauses[-1] if clauses else None)
            options = {k: [c for c in v if c.steps <= total - steps] for k, v in options.items()}
            options = {k: v for k, v in options.items() if v}
            if not options:
                break
            clause = rng.choice(options[rng.choice(sorted(options))])
            _apply(state, clause)
            clauses.append(clause)
            steps += clause.steps
        if steps == total and _has_effect(clauses):
            return clauses
    raise RuntimeError("could not compose a command")


def _has_effect(clauses: list[Clause]) -> bool:
    # 같은 대상을 연달아 두 번 보거나 가는 명령은 피한다.
    return all(not (a.action == b.action and a.obj == b.obj and a.action in ("move", "look")) for a, b in zip(clauses, clauses[1:]))


class Namer:
    """대상을 부르는 방식(label, 종류, 거리, 들고 있는 것, it)을 객체마다 한 번 정하고 패턴을 돌려준다."""

    def __init__(self, rng: random.Random, world: World, clauses: list[Clause]):
        self.rng = rng
        self.world = world
        mentioned: dict[str, set[str]] = {}
        for clause in clauses:
            for label in (clause.obj, clause.anchor):
                if label:
                    mentioned.setdefault(object_kind(label), set()).add(label)
        self.ambiguous = {kind for kind, labels in mentioned.items() if len(labels) > 1}
        first = {clause.obj for clause in clauses[:1]} | {clause.anchor for clause in clauses[:1]}
        self.style: dict[str, str] = {}
        for labels in mentioned.values():
            for label in sorted(labels):
                self.style[label] = self._choose(label, label in first)
        self.seen: set[str] = set()

    def _choose(self, label: str, first: bool) -> str:
        kind = object_kind(label)
        if label in self.world.carried:
            return "label" if kind in self.ambiguous or self.rng.random() < 0.4 else "held"
        if kind == "furnace":
            return self.rng.choice(["label", "furnace"])
        if kind in self.ambiguous:
            return "label"
        same = self.world.of_kind(kind)
        styles = ["label", "label", "type", "type"]
        if first and len(same) >= 2:
            nearest = self.world.nearest(same)
            farthest = max(same, key=self.world.distance)
            if label == nearest:
                styles.append("nearest")
            elif label == farthest:
                styles.append("farthest")
        return self.rng.choice(styles)

    def pattern(self, label: str) -> str:
        return object_kind(label) if self.style[label] == "type" else label

    def text(self, label: str, pronoun: bool = False) -> str:
        style = self.style[label]
        kind = object_kind(label)
        noun, indefinite = NOUNS[kind]
        again = label in self.seen
        self.seen.add(label)
        if pronoun:
            return "it"
        if style == "label":
            return label
        if style == "furnace":
            return "the furnace"
        if style == "held":
            return self.rng.choice(HELD).format(noun=noun)
        if again:
            return f"the {noun}"
        if style == "type":
            return self.rng.choice(indefinite)
        word = self.rng.choice(["nearest", "closest"]) if style == "nearest" else "farthest"
        return self.rng.choice([f"the {word} {noun}", f"the {noun} {word} from you" if style == "farthest" else f"the {noun} closest to you"])


def _render(rng: random.Random, world: World, clauses: list[Clause], setup: list[dict]) -> dict:
    namer = Namer(rng, world, clauses)
    parts, steps, reference = [], [], []
    previous: Clause | None = None
    for clause in clauses:
        pronoun = previous is not None and previous.action == "pick" and previous.obj == clause.obj and clause.action in ("drop", "drop_near", "store")
        x = namer.text(clause.obj, pronoun) if clause.obj else None
        if clause.action == "store" and not pronoun and clause.obj in world.carried and rng.random() < 0.5:
            x = "what you are holding"
        y = namer.text(clause.anchor) if clause.anchor else None
        template = rng.choice(TEMPLATES[clause.action])
        parts.append(template.replace("{X}", x or "").replace("{Y}", y or ""))
        px = namer.pattern(clause.obj) if clause.obj else None
        py = namer.pattern(clause.anchor) if clause.anchor else None
        if clause.action == "move":
            steps.append(_step("move_to", px)); reference.append(_act("move_to", clause.obj))
        elif clause.action == "look":
            steps.append(_step("look_at", px)); reference.append(_act("look_at", clause.obj))
        elif clause.action == "pick":
            steps.append(_step("pick", px)); reference.append(_act("pick", clause.obj))
        elif clause.action == "store":
            steps.append(_step("store")); reference.append(_act("store"))
        elif clause.action == "drop":
            steps.append(_step("drop", px)); reference.append(_act("drop", clause.obj))
        elif clause.action == "drop_near":
            steps.append(_step("drop", px, py)); reference.append(_act("drop", clause.obj, clause.anchor))
        else:  # carry
            steps += [_step("pick", px), _step("drop", px, py)]
            reference += [_act("pick", clause.obj), _act("drop", clause.obj, clause.anchor)]
        previous = clause

    if len(parts) == 1:
        command = parts[0]
    else:
        command = _clause(parts[0])
        for part in parts[1:]:
            command += rng.choice(JOIN) + _clause(part)
        command = _sentence(command) + "."
    return {"command": command, "steps": steps, "ordered": True, "setup": setup, "reference": reference,
            "goals": _goals(world, clauses, namer, setup)}


def _goals(world: World, clauses: list[Clause], namer: Namer, setup: list[dict]) -> list[dict]:
    state = State(world.holding, list(world.inventory))
    picked, stored, near, dropped = [], set(), [], []
    last_walk: Clause | None = None
    for clause in clauses:
        if clause.action in ("pick", "carry"):
            picked.append(clause.obj)
        if clause.action == "store":
            stored.add(clause.obj)
        if clause.action in ("drop_near", "carry"):
            near.append((clause.obj, clause.anchor))
        if clause.action == "drop":
            dropped.append(clause.obj)
        if clause.action in ("move", "pick", "drop_near", "carry"):
            last_walk = clause
        _apply(state, clause)
    carried_end = state.carried
    goals: list[dict] = [{"near": [namer.pattern(x), namer.pattern(y)]} for x, y in near]
    for label in dict.fromkeys(picked + sorted(stored)):
        if label not in carried_end:
            continue
        kind = "inventory" if label in stored and label in state.inventory else "carrying"
        goals.append({kind: namer.pattern(label)})
    for label in dropped:
        pattern = namer.pattern(label)
        if pattern == label or not any(object_kind(c) == pattern for c in carried_end):
            goals.append({"not_carrying": pattern})
    if last_walk is not None and last_walk.action == "move":
        goals.append({"agent_near": namer.pattern(last_walk.obj)})
    if not setup and all(c.action in ("move", "look") for c in clauses):
        goals.append({"empty_handed": True})
    return goals


def _setup_pick(rng: random.Random, world: World) -> list[dict]:
    return [_act("pick", rng.choice(world.pickable()))]


def single(rng: random.Random, observation: ObservationDTO, category: str) -> Draft:
    world = World(observation)
    setup = _setup_pick(rng, world) if category in ("drop", "store") else []

    def finalize(after: ObservationDTO) -> dict:
        w = World(after)
        if category in ("move", "look"):
            clauses = [Clause(category, rng.choice(w.free()))]
        elif category == "pick":
            clauses = [Clause("pick", rng.choice([x for x in w.pickable() if x not in w.carried]))]
        elif category == "store":
            clauses = [Clause("store", w.holding)]
        else:  # drop
            held = w.holding
            clauses = [Clause("drop", held)] if rng.random() < 0.5 else [Clause("drop_near", held, rng.choice(w.free()))]
        return _render(rng, w, clauses, setup)

    return Draft(setup, finalize)


def composite(rng: random.Random, observation: ObservationDTO, total_steps: Callable[[random.Random], int]) -> Draft:
    world = World(observation)
    setup = _setup_pick(rng, world) if rng.random() < 0.2 else []
    total = total_steps(rng)

    def finalize(after: ObservationDTO) -> dict:
        w = World(after)
        return _render(rng, w, _compose(rng, w, total), setup)

    return Draft(setup, finalize)


# ---------------- 추상적 명령 ----------------


def abstract(rng: random.Random, observation: ObservationDTO, category: str = "") -> Draft:
    world = World(observation)
    family = rng.choice(ABSTRACT)
    setup = _setup_pick(rng, world) if family is _drop_held else []

    def finalize(after: ObservationDTO) -> dict:
        task = family(rng, World(after))
        task.update(setup=setup, ordered=task.get("ordered", True))
        task["category"] = family.__name__.lstrip("_")
        return task

    return Draft(setup, finalize)


def _place(rng: random.Random, w: World, exclude_kind: str | None = None) -> tuple[str, str, str]:
    """내려놓을 자리: (문장 표현, 패턴, 정답 label)."""
    choices = [label for label in w.free() if object_kind(label) != exclude_kind]
    label = rng.choice(choices)
    kind = object_kind(label)
    if kind == "furnace":
        return "the furnace", label, label
    if rng.random() < 0.5:
        return label, label, label
    return rng.choice(NOUNS[kind][1]), kind, w.nearest(w.of_kind(kind))


def _pickable_free(w: World, exclude: set[str] = frozenset()) -> list[str]:
    return [x for x in w.pickable() if x not in w.carried and x not in exclude]


def _pick_any(rng, w):
    text = rng.choice(["Grab whatever you like.", "Pick something up.", "Take any one object.",
                       "Pick up an object of your choice.", "Grab one thing, anything is fine."])
    return {"command": text, "steps": [_step("pick", ANY)], "goals": [{"carrying": ANY}],
            "reference": [_act("pick", w.nearest(_pickable_free(w)))]}


def _move_any(rng, w):
    text = rng.choice(["Go somewhere.", "Wander over to something.", "Walk to any object you like.", "Go stand next to something."])
    return {"command": text, "steps": [_step("move_to", ANY)], "goals": [{"empty_handed": True}],
            "reference": [_act("move_to", w.nearest(w.free()))]}


def _look_any(rng, w):
    text = rng.choice(["Look at something.", "Look at anything you find interesting.", "Turn and face any object."])
    return {"command": text, "steps": [_step("look_at", ANY)], "goals": [{"empty_handed": True}],
            "reference": [_act("look_at", w.nearest(w.free()))]}


def _any_to_place(rng, w):
    text, pattern, anchor = _place(rng, w)
    command = rng.choice(["Put something next to {Y}.", "Bring any object to {Y}.", "Place an item of your choice beside {Y}.",
                          "Take whatever you like over to {Y}."]).format(Y=text)
    item = w.nearest(_pickable_free(w, {anchor}))
    return {"command": command, "steps": [_step("pick", ANY), _step("drop", ANY, pattern)],
            "goals": [{"near": [ANY, pattern]}], "reference": [_act("pick", item), _act("drop", item, anchor)]}


def _kind_to_place(rng, w):
    kind = rng.choice(sorted({object_kind(x) for x in _pickable_free(w)}))
    noun, indefinite = NOUNS[kind]
    text, pattern, anchor = _place(rng, w, exclude_kind=kind)
    command = rng.choice([f"Put any {noun} near {{Y}}.", f"Bring whichever {noun} you like to {{Y}}.",
                          f"Make sure there is {rng.choice(indefinite)} next to {{Y}}."]).format(Y=text)
    item = w.nearest(w.of_kind(kind))
    return {"command": command, "steps": [_step("pick", kind), _step("drop", kind, pattern)],
            "goals": [{"near": [kind, pattern]}], "reference": [_act("pick", item), _act("drop", item, anchor)]}


def _extreme(rng, w):
    kinds = sorted(k for k in {object_kind(x) for x in _pickable_free(w)} if len(w.of_kind(k)) >= 2)
    kind = rng.choice(kinds)
    noun = NOUNS[kind][0]
    near = rng.random() < 0.5
    label = w.nearest(w.of_kind(kind)) if near else max(w.of_kind(kind), key=w.distance)
    which = "closest" if near else "farthest"
    action = rng.choice(["move", "look", "pick"])
    text = {
        "move": [f"Walk to whichever {noun} is {which} to you.", f"Go to the {noun} that is {which} {'to' if near else 'from'} you."],
        "look": [f"Face the {which} {noun}.", f"Look at the {noun} that is {which} {'to' if near else 'from'} you."],
        "pick": [f"Pick up the {'nearest' if near else 'farthest'} {noun}.", f"Grab the {noun} {which} {'to' if near else 'from'} you."],
    }[action]
    tool = {"move": "move_to", "look": "look_at", "pick": "pick"}[action]
    goals = {"move": [{"agent_near": label}, {"empty_handed": True}], "look": [{"empty_handed": True}], "pick": [{"carrying": label}]}[action]
    return {"command": rng.choice(text), "steps": [_step(tool, label)], "goals": goals, "reference": [_act(tool, label)]}


def _store_any(rng, w):
    text = rng.choice(["Take an object of your choice and store it.", "Put something in your inventory.",
                       "Pick anything you like and keep it in your inventory."])
    item = w.nearest(_pickable_free(w))
    return {"command": text, "steps": [_step("pick", ANY), _step("store")], "goals": [{"inventory": ANY}],
            "reference": [_act("pick", item), _act("store")]}


def _drop_held(rng, w):
    text = rng.choice(["Get rid of what you're holding.", "Put down whatever you're carrying.", "Drop what you have.",
                       "Set down the thing in your hand."])
    return {"command": text, "steps": [_step("drop", w.holding)], "goals": [{"empty_handed": True}],
            "reference": [_act("drop", w.holding)]}


def _relocate_any(rng, w):
    text = rng.choice(["Move something somewhere else.", "Choose an item, carry it a bit, and set it down.",
                       "Pick up any object and put it down somewhere."])
    item = w.nearest(_pickable_free(w))
    return {"command": text, "steps": [_step("pick", ANY), _step("drop", ANY)], "goals": [{"empty_handed": True}],
            "reference": [_act("pick", item), _act("drop", item)]}


def _couple_to_place(rng, w):
    anchor = rng.choice(w.free())
    text = "the furnace" if object_kind(anchor) == "furnace" else anchor
    command = rng.choice(["Put a couple of things beside {Y}.", "Gather two objects next to {Y}.",
                          "Bring two items of your choice to {Y}."]).format(Y=text)
    items = sorted(_pickable_free(w, {anchor}), key=w.distance)[:2]
    return {"command": command, "ordered": False,
            "steps": [_step("pick", ANY), _step("pick", ANY), _step("drop", ANY, anchor), _step("drop", ANY, anchor)],
            "goals": [{"near": [ANY, anchor]}],
            "reference": [_act("pick", items[0]), _act("drop", items[0], anchor), _act("pick", items[1]), _act("drop", items[1], anchor)]}


def _something_near_something(rng, w):
    text = rng.choice(["Put something next to something else.", "Place any object beside any other object."])
    item = w.nearest(_pickable_free(w))
    anchor = w.nearest([x for x in w.free() if x != item])
    return {"command": text, "steps": [_step("pick", ANY), _step("drop", ANY, ANY)], "goals": [{"near": [ANY, ANY]}],
            "reference": [_act("pick", item), _act("drop", item, anchor)]}


def _hold_kind(rng, w):
    kind = rng.choice(sorted({object_kind(x) for x in _pickable_free(w)}))
    noun, indefinite = NOUNS[kind]
    text = rng.choice([f"Hold onto any {noun}.", f"Make sure you are carrying {rng.choice(indefinite)}.",
                       f"Get yourself {rng.choice(indefinite)}."])
    return {"command": text, "steps": [_step("pick", kind)], "goals": [{"carrying": kind}],
            "reference": [_act("pick", w.nearest(w.of_kind(kind)))]}


def _visit_kind_and_look(rng, w):
    kind = rng.choice(sorted({object_kind(x) for x in _pickable_free(w)}))
    noun = NOUNS[kind][0]
    target = rng.choice([x for x in w.free() if object_kind(x) != kind])
    text = "the furnace" if object_kind(target) == "furnace" else target
    command = rng.choice([f"Head toward any {noun} and look at {{Y}} from there.",
                          f"Go to whichever {noun} you like, then face {{Y}}."]).format(Y=text)
    visit = w.nearest(w.of_kind(kind))
    return {"command": command, "steps": [_step("move_to", kind), _step("look_at", target)],
            "goals": [{"agent_near": kind}, {"empty_handed": True}],
            "reference": [_act("move_to", visit), _act("look_at", target)]}


ABSTRACT = [_pick_any, _move_any, _look_any, _any_to_place, _kind_to_place, _extreme, _store_any, _drop_held,
            _relocate_any, _couple_to_place, _something_near_something, _hold_kind, _visit_kind_and_look]


def draft(generator: str, category: str, rng: random.Random, observation: ObservationDTO) -> Draft:
    if generator == "single":
        return single(rng, observation, category)
    if generator == "double":
        return composite(rng, observation, lambda r: 2)
    if generator == "complex":
        return composite(rng, observation, lambda r: r.choice([3, 3, 4, 4, 5]))
    if generator == "abstract":
        return abstract(rng, observation)
    raise ValueError(f"Unknown generator '{generator}'")
