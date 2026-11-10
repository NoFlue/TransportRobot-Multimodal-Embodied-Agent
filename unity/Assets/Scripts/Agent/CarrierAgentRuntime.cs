using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Unity.AI.Navigation;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace MultimodalAgent
{
    /// <summary>
    /// 행동 서버. Python 인지 서버의 요청을 받아 AgentAction을 하나씩 실행하고,
    /// 실행 피드백과 갱신된 관측을 돌려준다.
    /// </summary>
    [DefaultExecutionOrder(-100)]
    [RequireComponent(typeof(AgentCommandClient))]
    public class CarrierAgentRuntime : MonoBehaviour
    {
        [SerializeField] CarrierAgent agent;
        [SerializeField] Transform itemsRoot;
        [SerializeField] int port = 8001;
        [SerializeField] int imageSize = 512;
        [SerializeField] float actionTimeoutSeconds = 60f;
        [Tooltip("Components implementing IAgentCommandInput. Empty uses components on this GameObject.")]
        [SerializeField] MonoBehaviour[] commandInputs;

        readonly List<IAgentCommandInput> subscribedInputs = new List<IAgentCommandInput>();
        public AgentCommandClient Commands { get; private set; }

        static readonly Stopwatch Clock = Stopwatch.StartNew();

        readonly ConcurrentQueue<Action> mainThreadQueue = new ConcurrentQueue<Action>();
        readonly List<(LabeledObject obj, Transform parent, Vector3 position, Quaternion rotation)> initialObjects =
            new List<(LabeledObject, Transform, Vector3, Quaternion)>();
        // NavMesh를 마지막으로 구웠을 때 고정 객체(화로)의 자세. 달라지면 다시 굽는다.
        readonly Dictionary<LabeledObject, Pose> bakedFixedPoses = new Dictionary<LabeledObject, Pose>();

        AgentHttpServer server;
        readonly ObjectLocator locator = new ObjectLocator();
        Vector3 agentStartPosition;
        Quaternion agentStartRotation;

        AgentAction currentAction;
        Coroutine currentRoutine;
        TaskCompletionSource<ActionResultDto> currentCompletion;
        double currentStartedAt;
        bool cancelRequested;
        volatile bool busy;
        string instanceName;
        int processId;

        /// <summary>Unity 시계(초). Python은 /clock으로 자기 시계와의 차이를 보정한다.</summary>
        public static double Now => Clock.Elapsed.TotalSeconds;

        /// <summary>입력 인터페이스를 통해 받은 마지막 명령. 관측에 담긴다.</summary>
        public string LastMessage { get; private set; }

        /// <summary>HTTP 서버를 열지 못했을 때의 이유. 채팅 패널이 화면에 보여 준다.</summary>
        public string ServerError { get; private set; }

        public CarrierAgent Agent => agent;

        void Awake()
        {
            Commands = GetComponent<AgentCommandClient>();
            Application.runInBackground = true;
            foreach (var surface in FindObjectsByType<NavMeshSurface>(FindObjectsSortMode.None))
            {
                if (surface.navMeshData == null) surface.BuildNavMesh();
            }
        }

        void OnEnable()
        {
            var sources = commandInputs != null && commandInputs.Length > 0
                ? commandInputs : GetComponents<MonoBehaviour>();
            foreach (var source in sources)
            {
                if (source is not IAgentCommandInput input || subscribedInputs.Contains(input)) continue;
                input.CommandSubmitted += ReceiveCommand;
                subscribedInputs.Add(input);
            }
        }

        void OnDisable()
        {
            foreach (var input in subscribedInputs) input.CommandSubmitted -= ReceiveCommand;
            subscribedInputs.Clear();
        }

        void ReceiveCommand(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return;
            if (ServerError != null)
            {
                Commands.ReportError(ServerError);
                return;
            }
            LastMessage = text.Trim();
            Commands.Submit(LastMessage);
        }

        void Start()
        {
            instanceName = Application.isEditor ? "editor" : "player";
            processId = Process.GetCurrentProcess().Id;
            agent.EnableNavigation();
            agentStartPosition = agent.transform.position;
            agentStartRotation = agent.transform.rotation;
            foreach (var obj in LabeledObject.All)
            {
                initialObjects.Add((obj, obj.transform.parent, obj.transform.position, obj.transform.rotation));
                if (!obj.Pickable) bakedFixedPoses[obj] = new Pose(obj.transform.position, obj.transform.rotation);
            }

            server = new AgentHttpServer(HandleAsync);
            try
            {
                server.Start(port);
                Debug.Log($"CarrierAgentRuntime listening on http://127.0.0.1:{port}/");
            }
            catch (Exception e)
            {
                server.Dispose();
                server = null;
                ServerError = $"Port {port} is in use by another process (another Play session or a built player). " +
                              $"Python commands will not reach this scene. Close the other process and press Play again. ({e.Message})";
                Debug.LogError($"CarrierAgentRuntime could not listen on port {port}: {e.Message}");
            }
        }

        void Update()
        {
            while (mainThreadQueue.TryDequeue(out var work))
            {
                try
                {
                    work();
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                }
            }
        }

        void OnDestroy()
        {
            server?.Dispose();
            server = null;
            currentCompletion?.TrySetCanceled();
        }

        // ---------------- HTTP ----------------

        async Task<HttpReply> HandleAsync(string method, string path, string body)
        {
            switch (method + " " + path)
            {
                case "GET /health":
                    return Json(200, new { status = "ok", busy, instance = instanceName, pid = processId });
                case "GET /clock":
                    return Json(200, new { unity_time = Now });
                case "GET /observe":
                    return Json(200, await OnMainThread(BuildObservation));
                case "POST /perceive/view":
                {
                    var view = Parse<ViewRequestDto>(body) ?? new ViewRequestDto();
                    var image = await OnMainThread(() => Convert.ToBase64String(CaptureEyeImage(view.Yaw)));
                    return Json(200, new { yaw = view.Yaw, image_png_base64 = image });
                }
                case "POST /perceive/locate":
                {
                    var locate = Parse<LocateRequestDto>(body);
                    if (locate == null) return Json(400, new { error = "Body must be {yaw, x, y, name}." });
                    return Json(200, await OnMainThread(() => locator.Locate(agent.EyeCamera, locate)));
                }
                case "POST /act":
                    return await HandleActAsync(body);
                case "POST /act/cancel":
                    return Json(200, await OnMainThread(() => new { cancelled = RequestCancel() }));
                case "POST /world/reset":
                {
                    var reset = Parse<ResetRequestDto>(body);
                    await OnMainThread(() => ResetWorld(reset?.Seed));
                    return Json(200, new { status = "reset", seed = reset?.Seed });
                }
                default:
                    return Json(404, new { error = $"No endpoint {method} {path}" });
            }
        }

        async Task<HttpReply> HandleActAsync(string body)
        {
            ActRequestDto request;
            try
            {
                request = JsonConvert.DeserializeObject<ActRequestDto>(body);
            }
            catch (JsonException e)
            {
                return Json(400, new { error = $"Invalid JSON: {e.Message}" });
            }
            if (request == null || string.IsNullOrEmpty(request.Action))
            {
                return Json(400, new { error = "Field 'action' is required." });
            }

            var completion = new TaskCompletionSource<ActionResultDto>(TaskCreationOptions.RunContinuationsAsynchronously);
            var accepted = await OnMainThread(() => TryStartAction(request, completion));
            if (!accepted) return Json(409, new { error = "Another action is running." });
            return Json(200, await completion.Task);
        }

        static HttpReply Json(int status, object payload) => new HttpReply(status, JsonConvert.SerializeObject(payload));

        static T Parse<T>(string body) where T : class
        {
            if (string.IsNullOrWhiteSpace(body)) return null;
            try
            {
                return JsonConvert.DeserializeObject<T>(body);
            }
            catch (JsonException)
            {
                return null;
            }
        }

        /// <summary>몸은 그대로 두고 시점 카메라만 yaw만큼 돌린 채 작업한다(고개 돌리기).</summary>
        public static void WithYaw(Camera eye, float yaw, Action work)
        {
            var eyeTransform = eye.transform;
            var saved = eyeTransform.rotation;
            eyeTransform.rotation = Quaternion.AngleAxis(yaw, Vector3.up) * saved;
            try
            {
                work();
            }
            finally
            {
                eyeTransform.rotation = saved;
            }
        }

        Task<T> OnMainThread<T>(Func<T> work)
        {
            var completion = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
            mainThreadQueue.Enqueue(() =>
            {
                try
                {
                    completion.SetResult(work());
                }
                catch (Exception e)
                {
                    completion.SetException(e);
                }
            });
            return completion.Task;
        }

        Task<bool> OnMainThread(Action work) => OnMainThread(() =>
        {
            work();
            return true;
        });

        // ---------------- 행동 실행 ----------------

        bool TryStartAction(ActRequestDto request, TaskCompletionSource<ActionResultDto> completion)
        {
            if (busy) return false;
            busy = true;
            currentCompletion = completion;
            currentStartedAt = Now;
            cancelRequested = false;

            var action = CreateAction(request, out var error);
            if (action == null)
            {
                Finish(request.Action, false, error);
                return true;
            }
            currentAction = action;
            currentRoutine = StartCoroutine(RunAction(action));
            return true;
        }

        AgentAction CreateAction(ActRequestDto request, out string error)
        {
            error = null;
            var needsLabel = request.Action != "store";
            if (needsLabel && string.IsNullOrWhiteSpace(request.Label))
            {
                error = $"{request.Action} requires a label.";
                return null;
            }
            switch (request.Action)
            {
                case "move_to": return new MoveAction(request.Label);
                case "look_at": return new LookAction(request.Label);
                case "pick": return new PickAction(request.Label);
                case "store": return new StoreAction();
                case "drop": return new DropAction(request.Label, request.NearLabel, itemsRoot);
                default:
                    error = $"Unknown action '{request.Action}'.";
                    return null;
            }
        }

        IEnumerator RunAction(AgentAction action)
        {
            var result = new ActionResult();
            if (!action.Validate(agent, out var error))
            {
                Finish(action.Name, false, error);
                yield break;
            }

            currentStartedAt = Now;
            var routine = action.Execute(agent, result);
            var deadline = Time.realtimeSinceStartup + actionTimeoutSeconds;
            var stack = new Stack<IEnumerator>();
            stack.Push(routine);
            while (stack.Count > 0)
            {
                if (cancelRequested)
                {
                    action.Cancel(agent);
                    result.Fail("Cancelled.");
                    break;
                }
                if (Time.realtimeSinceStartup > deadline)
                {
                    action.Cancel(agent);
                    result.Fail($"Timed out after {actionTimeoutSeconds:0} s.");
                    break;
                }

                bool hasNext;
                try
                {
                    hasNext = stack.Peek().MoveNext();
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                    action.Cancel(agent);
                    result.Fail($"Action error: {e.Message}");
                    break;
                }
                if (!hasNext)
                {
                    stack.Pop();
                    continue;
                }
                // 중첩 코루틴(TurnTowards 등)은 같은 timeout 안에서 직접 진행한다.
                if (stack.Peek().Current is IEnumerator nested) stack.Push(nested);
                else yield return stack.Peek().Current;
            }
            Finish(action.Name, result.Success, result.Feedback);
        }

        void Finish(string actionName, bool success, string feedback)
        {
            var completion = currentCompletion;
            var dto = new ActionResultDto
            {
                Action = actionName,
                Success = success,
                Feedback = feedback,
                StartedAt = currentStartedAt,
                FinishedAt = Now,
                Observation = BuildObservation(),
            };
            currentAction = null;
            currentRoutine = null;
            currentCompletion = null;
            busy = false;
            completion?.TrySetResult(dto);
        }

        bool RequestCancel()
        {
            if (currentAction == null) return false;
            cancelRequested = true;
            return true;
        }

        void ResetWorld(int? seed)
        {
            if (currentRoutine != null)
            {
                StopCoroutine(currentRoutine);
                currentAction?.Cancel(agent);
                Finish(currentAction?.Name ?? "unknown", false, "Cancelled by world reset.");
            }
            agent.ClearCarrying();
            foreach (var (obj, parent, position, rotation) in initialObjects)
            {
                obj.PlaceAt(parent, position, rotation);
            }
            locator.Forget();
            if (seed.HasValue)
            {
                // Terrain과 장애물(Obstacles 아래, 예: label이 붙은 Minecart)은 그대로 두고 나머지 label 객체만 옮긴다.
                var tracked = initialObjects.Select(entry => entry.obj).Where(obj => obj != null).ToList();
                var obstacles = tracked.Where(IsObstacle).ToList();
                WorldRandomizer.Apply(seed.Value, agentStartPosition, tracked.Except(obstacles).ToList(),
                    obstacles.Select(obj => obj.transform.position), SyncNavMesh);
            }
            else
            {
                SyncNavMesh();
            }
            agent.Teleport(agentStartPosition, agentStartRotation);
            LastMessage = null;
        }

        /// <summary>장애물(Obstacles 아래)에 붙은 label 객체인지. 무작위 배치에서 옮기지 않는다.</summary>
        static bool IsObstacle(LabeledObject obj) =>
            obj.GetComponentsInParent<Transform>(true).Any(t => t.name == "Obstacles");

        /// <summary>고정 객체가 마지막으로 구운 자리에서 옮겨졌으면 NavMesh를 다시 굽는다.</summary>
        void SyncNavMesh()
        {
            Physics.SyncTransforms();
            var changed = bakedFixedPoses.Any(entry => entry.Key != null &&
                ((entry.Value.position - entry.Key.transform.position).sqrMagnitude > 1e-4f ||
                 Quaternion.Angle(entry.Value.rotation, entry.Key.transform.rotation) > 0.1f));
            if (!changed) return;
            foreach (var surface in FindObjectsByType<NavMeshSurface>(FindObjectsSortMode.None)) surface.BuildNavMesh();
            foreach (var obj in bakedFixedPoses.Keys.ToList())
            {
                if (obj != null) bakedFixedPoses[obj] = new Pose(obj.transform.position, obj.transform.rotation);
            }
        }

        // ---------------- 관측 ----------------

        ObservationDto BuildObservation()
        {
            var state = new AgentStateDto
            {
                Position = Vec3Dto.From(agent.Position),
                Holding = agent.Holding != null ? agent.Holding.Label : null,
                Inventory = agent.Inventory.Select(o => o.Label).ToList(),
            };
            return new ObservationDto
            {
                Timestamp = Now,
                Agent = state,
                Objects = LabeledObject.All
                    .OrderBy(o => o.Label, StringComparer.Ordinal)
                    .Select(o => ObjectDataDto.From(o.ToObjectData()))
                    .ToList(),
                Message = LastMessage,
                ImagePngBase64 = Convert.ToBase64String(CaptureEyeImage()),
            };
        }

        byte[] CaptureEyeImage(float yaw = 0f)
        {
            var eye = agent.EyeCamera;
            var target = RenderTexture.GetTemporary(imageSize, imageSize, 24, RenderTextureFormat.ARGB32);
            var previousTarget = eye.targetTexture;
            var previousActive = RenderTexture.active;
            var texture = new Texture2D(imageSize, imageSize, TextureFormat.RGB24, false);
            byte[] png = null;
            try
            {
                WithYaw(eye, yaw, () =>
                {
                    eye.targetTexture = target;
                    eye.Render();
                    RenderTexture.active = target;
                    texture.ReadPixels(new Rect(0, 0, imageSize, imageSize), 0, 0);
                    texture.Apply();
                    png = texture.EncodeToPNG();
                });
                return png;
            }
            finally
            {
                eye.targetTexture = previousTarget;
                RenderTexture.active = previousActive;
                RenderTexture.ReleaseTemporary(target);
                Destroy(texture);
            }
        }
    }
}
