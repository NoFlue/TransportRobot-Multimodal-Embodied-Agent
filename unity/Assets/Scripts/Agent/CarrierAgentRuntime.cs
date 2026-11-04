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
    public class CarrierAgentRuntime : MonoBehaviour
    {
        [SerializeField] CarrierAgent agent;
        [SerializeField] Transform itemsRoot;
        [SerializeField] int port = 8001;
        [SerializeField] int imageSize = 512;
        [SerializeField] float actionTimeoutSeconds = 60f;

        static readonly Stopwatch Clock = Stopwatch.StartNew();

        readonly ConcurrentQueue<Action> mainThreadQueue = new ConcurrentQueue<Action>();
        readonly List<(LabeledObject obj, Transform parent, Vector3 position, Quaternion rotation)> initialObjects =
            new List<(LabeledObject, Transform, Vector3, Quaternion)>();

        AgentHttpServer server;
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

        /// <summary>Unity 채팅에서 마지막으로 입력받은 메세지. 관측에 담긴다.</summary>
        public string LastMessage { get; set; }

        /// <summary>HTTP 서버를 열지 못했을 때의 이유. 채팅 패널이 화면에 보여 준다.</summary>
        public string ServerError { get; private set; }

        public CarrierAgent Agent => agent;

        void Awake()
        {
            Application.runInBackground = true;
            foreach (var surface in FindObjectsByType<NavMeshSurface>(FindObjectsSortMode.None))
            {
                if (surface.navMeshData == null) surface.BuildNavMesh();
            }
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
                case "POST /act":
                    return await HandleActAsync(body);
                case "POST /act/cancel":
                    return Json(200, await OnMainThread(() => new { cancelled = RequestCancel() }));
                case "POST /world/reset":
                    await OnMainThread(ResetWorld);
                    return Json(200, new { status = "reset" });
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

        void ResetWorld()
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
            agent.Teleport(agentStartPosition, agentStartRotation);
            LastMessage = null;
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

        byte[] CaptureEyeImage()
        {
            var eye = agent.EyeCamera;
            var target = RenderTexture.GetTemporary(imageSize, imageSize, 24, RenderTextureFormat.ARGB32);
            var previousTarget = eye.targetTexture;
            var previousActive = RenderTexture.active;
            var texture = new Texture2D(imageSize, imageSize, TextureFormat.RGB24, false);
            try
            {
                eye.targetTexture = target;
                eye.Render();
                RenderTexture.active = target;
                texture.ReadPixels(new Rect(0, 0, imageSize, imageSize), 0, 0);
                texture.Apply();
                return texture.EncodeToPNG();
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
