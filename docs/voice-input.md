# 음성·채팅 명령 입력

## 사용

1. Python 인지 서버를 실행하고 `Assets/Scenes/Simulation Scene.unity`를 Play한다.
2. 채팅은 기존 입력창의 `Send`, 음성은 `Voice (Korean)`으로 녹음을 시작한다.
3. 한국어 명령을 말한 뒤 `Stop & Send`를 누른다. 최대 30초이며, 종료 후 인식된 문장이 자동 전송된다.
4. 인식 문장은 음성 패널에, 명령과 실행 결과는 공통 채팅 창에 표시된다.

마이크가 없거나 권한이 차단되면 음성 패널에 오류를 표시한다. Windows 설정에서 데스크톱 앱의 마이크 접근을 허용해야 한다. 첫 음성 처리 때 모델을 로드한다. 실제 마이크 입력의 정확도는 장치·잡음·모델 크기에 따라 달라진다.

## 구조

`AgentChatPanel / WhisperCommandInput → IAgentCommandInput.CommandSubmitted → CarrierAgentRuntime → AgentCommandClient → Python Thread API`

- `IAgentCommandInput`은 최종 자연어 문장을 메인 스레드 이벤트로 제공한다.
- Runtime의 `commandInputs`에는 인터페이스 구현 컴포넌트를 지정한다. 비어 있으면 같은 GameObject에서 찾는다. 활성화 시 구독하고 비활성화 시 해제한다.
- Runtime이 `LastMessage`를 갱신하고 공통 클라이언트에 전달한다. 입력 구현은 Python API에 의존하지 않는다.
- `AgentCommandClient`가 대화 ID, 직렬 전송 큐, 폴링을 관리한다. 두 입력이 같은 대화를 사용한다. `New`는 전송 중 비활성화된다.
- Whisper는 녹음 종료 후 최종 결과만 한 번 전달한다. 무음·너무 짧은 입력과 비활성화 뒤 도착한 결과는 전달하지 않는다.
- 기존 씬을 재생성할 필요 없이 `Tools > Carrier Agent > Configure Command Inputs`로 입력 컴포넌트만 연결할 수 있다. 현재 Simulation Scene은 이미 연결되어 있다.

## 패키지·모델

- [whisper.unity](https://github.com/Macoron/whisper.unity), version `1.4.0`, commit `529a628a915a97799e89e061af9cb7c71407124d`를 UPM Git dependency로 고정했다.
- 같은 저장소의 다국어 `ggml-tiny.bin` (77,691,713 bytes)을 `unity/Assets/StreamingAssets/Whisper/`에 설치했다. SHA-256: `BE07E048E1E599AD46341C8D2A135645097A538221678B7ACDD1B1919C6E1B21`.
- 한국어 `ko`, 영어 번역 비활성화, 로컬 CPU 추론이 기본이다. STT 오디오는 외부 API로 보내지 않는다. 인식 텍스트는 기존 Python 명령 경로로 전송된다.
- 정확도를 높일 때는 whisper.cpp 호환 다국어 모델을 StreamingAssets에 넣고 `WhisperManager.ModelPath`를 변경한다. 영어 전용 `.en` 모델은 한국어 입력에 사용하지 않는다.
- 모델과 라이브러리 라이선스는 MIT이며 모델 옆에 라이선스를 보관했다.

## 검증

`CommandInputTests` EditMode 테스트는 텍스트 전달·구독 해제·중복 구독 방지, 무음·짧은 입력 무시, 행동 서버 오류 시 차단, 실제 Whisper 오디오 인식 결과의 공통 Runtime 큐 전달을 검증한다. 실제 인식 테스트는 저장소의 영어 음성 fixture를 사용하며 마이크나 실행 중인 Python 서버가 필요 없다.
