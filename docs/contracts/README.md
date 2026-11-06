# Unity ↔ Python 계약 예시

Unity 행동 서버(`CarrierAgentRuntime`, 기본 `http://127.0.0.1:8001`)와 Python 인지 서버가 주고받는 JSON 예시다. Python 테스트(`python-server/tests/test_contracts.py`)가 이 파일을 그대로 읽어 DTO를 검증한다.

| 파일 | Endpoint | 설명 |
|---|---|---|
| `observation.example.json` | `GET /observe` | 관측: 시각, 에이전트 상태, `ObjectData` 목록(label, 위치, pickable 속성), 입력받은 메세지, 512×512 PNG(예시는 1×1) |
| `act_request.example.json` | `POST /act` 요청 | 도구 호출. `action`은 move_to, look_at, pick, store, drop |
| `action_result.example.json` | `POST /act` 응답 | 성공 여부, 실행 피드백, Unity 시계 기준 실제 시작·종료 시각, 실행 후 관측 |

그 밖의 endpoint: `GET /health` → `{"status": "ok", "busy": false}`, `GET /clock` → `{"unity_time": 12.3}`, `POST /act/cancel` → `{"cancelled": true}`, `POST /world/reset` → `{"status": "reset"}`.

## 등록되지 않은 객체 찾기

명령에 나온 객체가 `ObjectData` 목록에 없으면 Python이 고개를 돌린 시점 이미지에서 VLM으로 찾고, Unity가 그 좌표로 객체를 특정해 등록한다. 등록된 객체는 다음 `/observe`부터 목록에 나온다. `/world/reset`은 이렇게 등록한 객체를 다시 지운다.

| Endpoint | 요청 | 응답 |
|---|---|---|
| `POST /perceive/view` | `{"yaw": 60}` (몸 정면 기준 도, 오른쪽 +) | `{"yaw": 60, "image_png_base64": "..."}` |
| `POST /perceive/locate` | `{"yaw": 60, "x": 0.48, "y": 0.61, "name": "barrel"}` (x·y는 0~1, 왼쪽 위 기준) | `{"found": true, "known": false, "feedback": "Registered Wooden Mining Barrel as barrel_1.", "object": {"label": "barrel_1", "position": {...}, "pickable": false}}` |

`known: true`는 이미 등록된 객체를 짚은 경우다. 지면만 맞으면 `found: false`다.
