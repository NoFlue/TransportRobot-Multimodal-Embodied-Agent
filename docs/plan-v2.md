# Multimodal Agent 재기획안 v2 (초안)

상태: Draft 0.2 · 작성일 2026-11-05

이전 구현(`archive/2026-11-05-reset/`)은 미리 정의한 시나리오만 실행할 수 있는 구조여서 폐기했다. 이 문서는 처음부터 다시 세운 기획이다.

## 1. 프로젝트 목표

이 프로젝트는 **미리 학습된 VLM(Vision-Language Model)** 을 그대로 사용해 Unity 3차원 환경에서 자연어 명령을 수행하는 **Multimodal Embodied Agent**를 구현한다. 모델을 추가로 학습시키지 않고, 환경 정보를 정리해서 제공하는 방식과 도구 설계로 성공률을 높인다.

- **ORA loop**: 에이전트는 Observation → Reasoning → Act 루프를 반복한다. 행동은 **Tool Calling**으로 Unity 안의 도구를 호출하는 방식으로만 일어난다.
- **구조화된 환경 정보**: 3차원 환경 객체의 속성과 관계를 구조화된 텍스트로 만들어 관측 이미지와 함께 에이전트에게 준다. 에이전트는 이 정보에 근거해 멀티모달 추론을 하고, 도구로 환경과 상호작용한다.
- 역할은 세 구성 요소로 나눈다.

| 구성 요소 | 구현 | 역할 |
|---|---|---|
| 인지 서버 | Python (LangGraph) | 관측·추론·행동 루프 조정. 프롬프트 구성, 기억 갱신, 도구 호출 관리, 측정 기록 |
| VLM 추론 서버 | 미리 학습된 VLM (OpenRouter 또는 로컬 서버) | 추론과 행동(호출할 도구) 선택 |
| 행동 서버 | Unity (`CarrierAgentRuntime`) | 도구 실행, 환경 갱신, 관측과 실행 피드백 반환 |

한 단계는 다음 순서로 진행하며, 에이전트가 명령을 완료했다고 판단할 때까지 반복한다.

1. 인지 서버가 행동 서버에서 관측(시점 이미지, 자신의 위치, 객체 목록)을 받는다.
2. 인지 서버가 관측을 구조화된 텍스트로 바꾸고, 대화 기억과 합쳐 프롬프트를 구성한 뒤 VLM 추론 서버에 보낸다.
3. VLM이 추론하고 호출할 도구와 인자를 고른다. 도구를 고르지 않고 답하면 명령 처리가 끝난다.
4. 인지 서버가 도구 호출을 행동 서버에 전달하고, `CarrierAgent`가 행동을 실행한다.
5. 행동 서버가 바뀐 환경을 갱신하고 실행 피드백과 새 관측을 돌려준다. 인지 서버는 기억을 갱신하고 1로 돌아간다.

## 2. 시뮬레이션 환경

- **씬**: Terrain으로 만든 울퉁불퉁한 길과 여러 장애물이 있다. 에이전트는 장애물을 피해 이동해야 한다.
- **객체**: 맵에 석탄(coal), 철(iron), 화로(furnace)가 있다. 석탄과 철은 여러 개일 수 있다.
- **판단 주체**: 에이전트는 명령을 받으면 스스로 판단해 어떤 행동을 할지 정한다. 명령별 정답 행동 순서는 미리 정하지 않는다.
- **대표 명령**: "석탄 1개와 철 1개를 주워서 화로 주변에 둬". 예상되는 흐름은 관측 → 석탄으로 이동 → 줍기 → 철로 이동 → 줍기 → 화로로 이동 → 내려놓기 → 완료 응답이다. 이 흐름은 설명용이며 코드에 넣지 않는다. 석탄과 철을 함께 운반하려면 에이전트가 두 개 이상을 들 수 있어야 한다(Store 행동과 인벤토리).

## 3. 확정된 요구사항

| 항목 | 결정 |
|---|---|
| 주도권 | Python 인지 서버가 관측을 보고 VLM에 판단을 맡긴 뒤 Unity에 행동을 명령한다. Unity는 세계 상태와 행동 실행을 책임진다. |
| 명령 입력 | **Unity 화면 안의 채팅 입력**과 **Python(CLI·HTTP)** 양쪽. 두 경로는 같은 처리 경로로 들어간다. |
| 대화 방식 | **대화형(thread)**. 같은 에이전트에게 명령을 이어서 보낼 수 있고, 이전 대화와 행동 맥락이 유지된다. |
| 추론 구조 | LangGraph 기반 Tool Calling 에이전트, ORA(Observation → Reasoning → Act) 루프. 추론과 행동 선택은 미리 학습된 VLM에 위임한다. |
| 환경 정보 | Unity의 `ObjectData`(label, 위치)를 받아 객체의 속성과 관계를 구조화된 텍스트로 만들어 관측 이미지와 함께 제공한다. |
| 모델 설정 | config로 모델 프로필을 고르고, OpenRouter API key는 별도 파일에 둔다. |
| 측정 | **단계마다** ① 추론 시작 → Unity에서 도구가 실제로 호출(실행 시작)되기까지 걸린 시간, ② 도구 실행 성공 여부. **명령마다** ③ 명령을 받은 시점 → 첫 행동이 실제로 시작된 시점 시간, ④ 최종 성공 여부. |
| 성공 판정 | 단계 성공은 Unity의 실행 결과로, 명령 성공은 처리가 끝난 뒤 **LLM이 최종 관측을 보고 판정**한다. |

## 4. 이전 구현의 실패에서 가져온 원칙

1. **자연어가 1급 입력이다.** 명령은 자유 문장이다. 시나리오, 정답 목표, 키워드 파서는 명령을 실행하는 경로에 두지 않는다.
2. **평가 정보는 실행의 전제 조건이 아니다.** 성공 판정은 실행이 끝난 뒤 따로 붙는다.
3. **경계 계약을 먼저 고정하고, 양쪽이 같은 JSON 예시로 테스트한다.**
4. **넓히기 전에 끝까지 잇는다.** 첫 단계에서 실제 Unity와 Python을 연결해 명령 하나가 행동까지 이어지는 최소 경로를 먼저 검증한다.
5. **Unity 테스트는 테스트 전용 asmdef로 분리한다.** player 빌드가 테스트 코드 때문에 깨지지 않게 한다.

## 5. 전체 구조

```mermaid
flowchart LR
    UI[Unity 채팅 UI] -->|POST /threads/{id}/commands| P
    CLI[Python CLI / HTTP 클라이언트] -->|같은 API| P
    subgraph P[Python 인지 서버]
        T[Thread 관리<br/>대화 기억·명령 큐] --> G[LangGraph ORA loop<br/>프롬프트 구성·기억 갱신·도구 호출]
        G --> J[LLM 판정]
        G --> R[측정 기록]
    end
    G <-->|프롬프트 / tool call| V[VLM 추론 서버<br/>미리 학습된 VLM]
    G -->|/observe, /act| E[Unity 행동 서버<br/>CarrierAgentRuntime]
    E --> A[CarrierAgent]
    J -->|/observe| E
    UI -->|GET /threads/{id} 폴링| P
```

- Unity → Python은 명령을 제출하고 대화 상태를 보여 주는 데만 쓴다.
- Python → VLM은 추론과 도구 선택에 쓴다.
- Python → Unity는 관측과 행동에 쓴다.
- Unity 세계 상태는 thread 안에서 명령이 바뀌어도 유지된다. 초기화는 명시적으로 요청할 때만 한다.

## 6. Python 인지 서버

**Thread 모델**

- thread = 하나의 대화. 사람 메시지, 에이전트 응답, 행동 기록이 쌓인다. LangGraph checkpointer에 thread_id로 저장한다.
- command = thread 안의 사람 명령 하나. 상태는 `queued → running → judging → done | failed | cancelled`.
- Unity 에이전트가 하나이므로 행동을 실행하는 명령은 전체에서 동시에 하나만 돈다. 나머지는 대기열에 둔다.

**API (초안)**

| Method | Path | 설명 |
|---|---|---|
| POST | `/threads` | 새 대화 |
| POST | `/threads/{id}/commands` | `{"text": "석탄 1개와 철 1개를 주워서 화로 주변에 둬"}` 제출 → command_id |
| GET | `/threads/{id}` | 메시지·명령 상태·에이전트 응답 (Unity UI가 폴링) |
| POST | `/commands/{id}/cancel` | 진행 중인 명령 취소 |
| GET | `/health` | 서버, VLM 추론 서버, Unity 연결 상태 |

CLI: `python -m agent chat`는 대화형 REPL이고 같은 서비스를 쓴다. `python -m agent serve`는 HTTP 서버를 띄운다.

**LangGraph Tool Calling 에이전트**

- 그래프는 `observe → reason → act → observe …` 순환이다. `reason` 노드는 도구 스키마를 bind한 VLM을 호출하고, `act` 노드는 VLM이 고른 tool call을 Unity `POST /act`로 보낸다.
- 자연어 명령은 VLM이 직접 해석해 도구를 고른다. Python 쪽에 키워드 파서나 규칙 기반 해석을 두지 않는다.
- 도구는 Unity의 `AgentAction`과 1:1로 대응하고, 대상은 `ObjectData.label`로 지정한다.

| Tool | 인자 | Unity 행동 |
|---|---|---|
| `move_to` | `label` | `MoveAction` — 대상 근처로 이동 |
| `look_at` | `label` | `LookAction` — 대상을 향해 시점 회전 |
| `pick` | `label` | `PickAction` — 가까이 있는 대상을 손에 듦 |
| `store` | 없음 | `StoreAction` — 손에 든 물건을 인벤토리에 넣음 |
| `drop` | `label`, `near_label`(선택) | `DropAction` — 물건을 내려놓음. `near_label`이 있으면 그 객체 주변에 둠 |

- 단계 수와 시간 한도가 있다. 모델이 tool 없이 사용자에게 답하면 명령 처리가 끝난다.

**프롬프트 구성과 기억 갱신**

- 프롬프트 = 시스템 지시(역할, 도구 사용 규칙) + 대화 기억 + 현재 관측(이미지 + 구조화된 텍스트) + 직전 도구의 실행 피드백.
- 기억은 thread의 메시지와 행동 기록이다. 이전 단계의 이미지는 컨텍스트 길이를 줄이기 위해 프롬프트에서 빼고, 그 단계의 행동과 결과 텍스트만 남긴다. 이미지는 최신 관측 하나만 보낸다.
- 구조화된 텍스트는 `ObservationDTO`에서 만든다. 속성은 label, 종류(label 접두어, 예: `coal_1` → coal), 위치, 에이전트와의 거리이고, 관계는 객체 사이 거리와 "근처(near)" 여부(반경은 config)다.

```text
[agent] position=(12.0, 0.5, 3.1) holding=[] inventory=[]
[objects]
- coal_1   type=coal    position=(15.2, 0.4, 8.0)  distance=5.8
- iron_1   type=iron    position=(4.1, 0.9, 10.3)  distance=10.7
- furnace_1 type=furnace position=(20.0, 0.2, 2.0) distance=8.1
[relations]
- coal_1 near furnace_1: no (distance 7.7)
- iron_1 near furnace_1: no (distance 17.9)
```

**DTO (Pydantic)**

Unity에서 받는 데이터는 아래 DTO로 받는다. 필드 이름은 Unity 쪽 JSON과 같게 맞춘다.

```python
class Vec3(BaseModel):
    x: float
    y: float
    z: float

class ObjectDataDTO(BaseModel):     # Unity ObjectData와 1:1
    label: str                       # 예: "coal_1"
    position: Vec3                   # 오브젝트 위치

class AgentStateDTO(BaseModel):
    position: Vec3                   # 자신의 위치
    holding: str | None = None       # 손에 든 객체 label
    inventory: list[str] = []        # 인벤토리에 든 객체 label

class ObservationDTO(BaseModel):
    timestamp: float                 # 관측 시각 (Unity 시계)
    agent: AgentStateDTO
    objects: list[ObjectDataDTO]
    message: str | None = None       # 입력받은 메세지 (Unity 채팅에서 온 명령)
    image_png_base64: str            # 관측한 시점 이미지 (512×512)

class ActionResultDTO(BaseModel):
    action: str
    success: bool
    feedback: str                    # 실행 피드백, 실패 이유
    started_at: float                # Unity에서 실제 실행을 시작한 시각
    finished_at: float
    observation: ObservationDTO      # 실행 후 갱신된 관측
```

**LLM 판정**

명령 처리가 끝나면 최종 관측(이미지 + 구조화된 텍스트)과 명령 문장을 판정 모델에 주고 `{success: bool, rationale: str}`를 받는다. 판정 모델은 행동 모델과 다른 프로필로 지정할 수 있다(자기 채점 편향을 줄이기 위해).

**측정**

- 두 프로세스의 시계가 다르므로, Unity의 `/clock` 응답과 요청 왕복 시간으로 시계 차이를 보정한다(로컬 loopback이라 오차는 왕복 시간의 절반, 수 ms 이하).
- 단계 기록 `steps.csv`: 단계마다 한 줄. thread_id, command_id, step, tool_name, tool_args, reasoning_seconds(VLM 요청 → 응답), step_to_call_seconds(단계 시작 → Unity가 보고한 실제 실행 시작), execution_seconds(실행 시작 → 종료), tool_success, feedback.
- 명령 기록 `results.csv`: 명령마다 한 줄. thread_id, command_id, text, model_profile, command_to_first_action_seconds(`t0` = Python이 명령을 받은 시각 → `t1` = 첫 실제 행동 시작 시각), judge_success, judge_rationale, steps, tool_success_rate, end_reason.

## 7. Unity 행동 서버

- **시나리오·정답 목표·키워드 명령 해석이 없다.** 세계 상태 제공과 행동 실행만 맡는다.

**클래스 설계**

| 클래스 | 역할 |
|---|---|
| `CarrierAgent` | 실제 움직임을 수행하는 에이전트(MonoBehaviour). Terrain 위 이동, 회전, 집기, 보관, 내려놓기와 손·인벤토리 상태를 가진다. |
| `CarrierAgentRuntime` | 행동 서버. HTTP 요청을 받아 `AgentAction`을 만들고 `CarrierAgent`에게 하나씩 실행시킨 뒤 실행 결과와 새 관측을 돌려준다. 관측 이미지 촬영과 `ObjectData` 목록 수집도 맡는다. |
| `AgentAction` | 모든 행동의 추상 클래스. 행동마다 이 클래스를 구체화한다. |
| `ObjectData` | 객체를 특정하는 `label`과 위치값. 관측마다 에이전트에게 보내 대상 지정의 정확도와 성공률을 높인다. |

```csharp
public abstract class AgentAction
{
    public abstract string Name { get; }

    // 실행 전 검사: 대상이 있는지, 거리가 충분한지 등. 실패하면 이유를 돌려준다.
    public abstract bool Validate(CarrierAgent agent, out string error);

    // 여러 프레임에 걸쳐 진행되므로 코루틴으로 실행하고 결과를 채운다.
    public abstract IEnumerator Execute(CarrierAgent agent, ActionResult result);
}

// 구체 클래스: MoveAction, LookAction, PickAction, StoreAction, DropAction

[Serializable]
public class ObjectData
{
    public string label;      // 씬 안에서 유일. 예: "coal_1", "iron_1", "furnace_1"
    public Vector3 position;
}
```

**Endpoint와 관측**

- Endpoint (초안): `GET /health`, `GET /clock`, `GET /observe`, `POST /act`, `POST /act/cancel`, `POST /world/reset`.
- 행동: Move, Look, Pick, Store, Drop. 각각 `AgentAction` 구체 클래스이고, 대상은 `ObjectData.label`로 지정한다.
- 관측: 512×512 시점 이미지, 에이전트 상태(위치, 손, 인벤토리), `ObjectData` 목록, 입력받은 메세지. Python `ObservationDTO`와 같은 JSON 형식이다.
- 실행 결과: 성공 여부, 실행 피드백(실패 이유), 실제 실행 시작·종료 시각, 실행 후 관측.
- 안전: 행동 하나씩 실행, 행동 timeout, 안전 정지, Python 연결이 끊기면 정지.
- 채팅 UI: 입력창 → Python `POST /threads/{id}/commands`, 대화 내용은 `GET /threads/{id}` 폴링으로 표시.

## 8. 진행 단계

| 단계 | 목표 | 완료 기준 |
|---|---|---|
| 0 | 계약 확정: Thread API, Env API, DTO와 공통 JSON 예시 | 문서와 예시 JSON 확정 |
| 1 | **최소 연결 경로**: `CarrierAgent`, `CarrierAgentRuntime`, `AgentAction`, `ObjectData` 골격과 observe, `MoveAction` 하나. Python의 thread, LangGraph ORA loop, CLI 명령 | 실제 Unity에서 CLI 자연어 명령 → 이동 성공 |
| 2 | 나머지 `AgentAction`(Look, Pick, Store, Drop), 구조화된 텍스트, Unity 채팅 UI, 대화 맥락 유지 | Unity 화면에서 "석탄 1개와 철 1개를 주워서 화로 주변에 둬" 수행, 이어서 명령 하나 더 실행 |
| 3 | 측정(시계 보정, 단계·명령 기록)과 LLM 판정 | `steps.csv`, `results.csv`에 시간·성공 여부 기록 |
| 4 | 테스트 정리, 테스트 asmdef 분리, player 빌드 | Unity·Python 테스트 통과, player 빌드 성공 |

## 9. 결정이 필요한 사항

1. **Unity 범용 모듈 재사용**: archive에 있는 행동(Actions), 내비게이션, 객체 레지스트리, 관측(ObservationProvider) 코드는 시나리오와 무관한 범용 부분이다. 검토한 뒤 `AgentAction`·`CarrierAgent` 구조에 맞게 가져다 쓸지, Unity도 완전히 새로 작성할지. Terrain 위 장애물 회피 이동을 Unity NavMesh로 할지도 함께 정한다.
2. **추론 형식과 기본 VLM**: 오늘 01:03에 이 대화 밖에서 들어간 변경이 ADAMO 방식(본문의 `[REASONING_NOTE]`)과 로컬 `qwen3.5:4b`(Ollama)였다. 이것을 기본으로 할지.
3. **판정 모델**: 행동 모델과 같은 모델을 쓸지, 별도 모델을 쓸지.
