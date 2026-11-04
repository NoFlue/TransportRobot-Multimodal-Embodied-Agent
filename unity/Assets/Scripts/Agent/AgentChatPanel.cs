using System.Collections;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

namespace MultimodalAgent
{
    /// <summary>
    /// 화면 안의 채팅 입력. 명령은 Python 인지 서버의 thread API로 보내고, 대화는 폴링으로 표시한다.
    /// </summary>
    public class AgentChatPanel : MonoBehaviour
    {
        [SerializeField] CarrierAgentRuntime runtime;
        [SerializeField] string pythonBaseUrl = "http://127.0.0.1:8000";
        [SerializeField] float pollSeconds = 1f;

        readonly List<string> lines = new List<string>();
        string threadId;
        string input = "";
        string status = "";
        Vector2 scroll;
        int shownMessageCount;
        bool sending;
        GUIStyle wrapStyle;

        void Start() => StartCoroutine(PollLoop());

        void OnGUI()
        {
            wrapStyle ??= new GUIStyle(GUI.skin.label) { wordWrap = true, richText = true };
            const float width = 480f;
            const float height = 300f;
            GUILayout.BeginArea(new Rect(10f, Screen.height - height - 10f, width, height), GUI.skin.box);
            GUILayout.Label(threadId == null ? "Agent chat (new conversation)" : $"Agent chat · thread {threadId.Substring(0, 8)}");

            scroll = GUILayout.BeginScrollView(scroll, GUILayout.ExpandHeight(true));
            foreach (var line in lines) GUILayout.Label(line, wrapStyle);
            GUILayout.EndScrollView();

            if (!string.IsNullOrEmpty(status)) GUILayout.Label(status, wrapStyle);

            var serverError = runtime != null ? runtime.ServerError : null;
            if (serverError != null)
            {
                GUILayout.Label($"<color=#ff6b6b><b>행동 서버를 시작하지 못해 명령을 보낼 수 없습니다.</b> {serverError}</color>", wrapStyle);
            }

            var submit = Event.current.type == EventType.KeyDown &&
                         (Event.current.keyCode == KeyCode.Return || Event.current.keyCode == KeyCode.KeypadEnter) &&
                         GUI.GetNameOfFocusedControl() == "chat_input";
            GUILayout.BeginHorizontal();
            GUI.SetNextControlName("chat_input");
            input = GUILayout.TextField(input, GUILayout.ExpandWidth(true));
            if (GUILayout.Button("Send", GUILayout.Width(60f))) submit = true;
            if (GUILayout.Button("New", GUILayout.Width(50f)))
            {
                threadId = null;
                lines.Clear();
                shownMessageCount = 0;
                status = "";
            }
            GUILayout.EndHorizontal();
            GUILayout.EndArea();

            if (submit && serverError == null && !sending && !string.IsNullOrWhiteSpace(input))
            {
                StartCoroutine(Send(input.Trim()));
                input = "";
                Event.current.Use();
            }
        }

        IEnumerator Send(string text)
        {
            sending = true;
            if (runtime != null) runtime.LastMessage = text;
            if (threadId == null)
            {
                string created = null;
                yield return Request("POST", "/threads", "{}", body => created = body);
                if (created == null)
                {
                    sending = false;
                    yield break;
                }
                threadId = JObject.Parse(created).Value<string>("thread_id");
            }
            var payload = JsonConvert.SerializeObject(new { text });
            yield return Request("POST", $"/threads/{threadId}/commands", payload, _ => { });
            sending = false;
        }

        IEnumerator PollLoop()
        {
            var wait = new WaitForSecondsRealtime(pollSeconds);
            while (true)
            {
                if (threadId != null)
                {
                    yield return Request("GET", $"/threads/{threadId}", null, ShowThread);
                }
                yield return wait;
            }
        }

        void ShowThread(string body)
        {
            var thread = JObject.Parse(body);
            var messages = thread["messages"] as JArray ?? new JArray();
            if (messages.Count != shownMessageCount)
            {
                lines.Clear();
                foreach (var message in messages)
                {
                    lines.Add(Format(message.Value<string>("role"), message.Value<string>("text")));
                }
                shownMessageCount = messages.Count;
                scroll.y = float.MaxValue;
            }
            var commands = thread["commands"] as JArray;
            if (commands != null && commands.Count > 0)
            {
                var last = commands[commands.Count - 1];
                status = $"command {last.Value<string>("status")}";
            }
        }

        static string Format(string role, string text)
        {
            switch (role)
            {
                case "user": return $"<b>you</b>: {text}";
                case "agent": return $"<color=#9fe39f><b>agent</b></color>: {text}";
                case "action": return $"<color=#a0c4ff>{text}</color>";
                default: return $"<i>{text}</i>";
            }
        }

        IEnumerator Request(string method, string path, string json, System.Action<string> onSuccess)
        {
            using (var request = new UnityWebRequest(pythonBaseUrl + path, method))
            {
                if (json != null)
                {
                    request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
                    request.SetRequestHeader("Content-Type", "application/json");
                }
                request.downloadHandler = new DownloadHandlerBuffer();
                request.timeout = 10;
                yield return request.SendWebRequest();
                if (request.result == UnityWebRequest.Result.Success)
                {
                    onSuccess(request.downloadHandler.text);
                }
                else if (request.responseCode == 404 && path.StartsWith("/threads/"))
                {
                    // Python 서버가 재시작되면 이전 thread가 없다. 다음 전송 때 새 대화를 만든다.
                    threadId = null;
                    shownMessageCount = 0;
                    lines.Clear();
                    status = "Python 서버가 재시작되어 대화가 초기화되었습니다. 명령을 다시 보내 주세요.";
                }
                else
                {
                    status = $"Python server error: {request.error}";
                }
            }
        }
    }
}
