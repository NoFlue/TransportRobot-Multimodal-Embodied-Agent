using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

namespace MultimodalAgent
{
    /// <summary>모든 입력이 공유하는 Python 대화·전송 큐.</summary>
    public class AgentCommandClient : MonoBehaviour
    {
        [SerializeField] string pythonBaseUrl = "http://127.0.0.1:8000";
        [SerializeField] float pollSeconds = 1f;
        readonly Queue<string> pending = new Queue<string>();
        readonly List<string> lines = new List<string>();
        string threadId;
        bool sending;
        int generation;
        public string ThreadId => threadId;
        public IReadOnlyList<string> Lines => lines;
        public string Status { get; private set; } = "";
        public bool IsSending => sending || pending.Count > 0;

        void Start() => StartCoroutine(PollLoop());

        public void Submit(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return;
            pending.Enqueue(text.Trim());
            if (!sending) StartCoroutine(SendPending());
        }

        public void ReportError(string message) => Status = message;

        public void NewConversation()
        {
            if (IsSending) return;
            generation++;
            threadId = null;
            lines.Clear();
            Status = "";
        }

        IEnumerator SendPending()
        {
            sending = true;
            try
            {
                while (pending.Count > 0)
                {
                    var text = pending.Dequeue();
                    if (threadId == null)
                    {
                        string created = null;
                        yield return Request("POST", "/threads", "{}", body => created = body);
                        if (created == null)
                        {
                            lines.Add($"<b>전송 실패</b>: {Escape(text)}");
                            continue;
                        }
                        threadId = JObject.Parse(created).Value<string>("thread_id");
                        if (string.IsNullOrEmpty(threadId))
                        {
                            ReportError("Python response has no thread_id. 명령을 다시 보내 주세요.");
                            continue;
                        }
                    }
                    bool sent = false;
                    yield return Request("POST", $"/threads/{threadId}/commands",
                        JsonConvert.SerializeObject(new { text }), _ => sent = true);
                    if (!sent) lines.Add($"<b>전송 실패 — 다시 입력해 주세요</b>: {Escape(text)}");
                }
            }
            finally { sending = false; }
        }

        IEnumerator PollLoop()
        {
            while (true)
            {
                if (threadId != null)
                    yield return Request("GET", $"/threads/{threadId}", null, ShowThread);
                yield return new WaitForSecondsRealtime(Mathf.Max(0.1f, pollSeconds));
            }
        }

        void ShowThread(string body)
        {
            var thread = JObject.Parse(body);
            var messages = thread["messages"] as JArray ?? new JArray();
            lines.Clear();
            foreach (var message in messages)
                lines.Add(Format(message.Value<string>("role"), message.Value<string>("text")));
            var commands = thread["commands"] as JArray;
            if (commands != null && commands.Count > 0)
                Status = $"command {commands[commands.Count - 1].Value<string>("status")}";
        }

        static string Escape(string text) => (text ?? "").Replace("<", "&lt;").Replace(">", "&gt;");
        static string Format(string role, string text)
        {
            text = Escape(text);
            switch (role)
            {
                case "user": return $"<b>you</b>: {text}";
                case "agent": return $"<color=#9fe39f><b>agent</b></color>: {text}";
                case "action": return $"<color=#a0c4ff>{text}</color>";
                default: return $"<i>{text}</i>";
            }
        }

        IEnumerator Request(string method, string path, string json, Action<string> onSuccess)
        {
            int requestedGeneration = generation;
            using (var request = new UnityWebRequest(pythonBaseUrl.TrimEnd('/') + path, method))
            {
                if (json != null)
                {
                    request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
                    request.SetRequestHeader("Content-Type", "application/json");
                }
                request.downloadHandler = new DownloadHandlerBuffer();
                request.timeout = 10;
                yield return request.SendWebRequest();
                if (requestedGeneration != generation) yield break;
                if (request.result == UnityWebRequest.Result.Success)
                {
                    try { onSuccess(request.downloadHandler.text); }
                    catch (Exception e) { Status = $"Invalid Python response: {e.Message}"; }
                }
                else if (request.responseCode == 404 && path.StartsWith("/threads/"))
                {
                    generation++;
                    threadId = null;
                    Status = "Python 서버의 대화가 초기화되었습니다. 실패한 명령을 다시 보내 주세요.";
                }
                else Status = $"Python server error: {request.error}";
            }
        }
    }
}
