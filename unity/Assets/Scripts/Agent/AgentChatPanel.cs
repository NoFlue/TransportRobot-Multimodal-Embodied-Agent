using System;
using UnityEngine;

namespace MultimodalAgent
{
    /// <summary>텍스트 입력과 대화 표시. 명령 전송은 Runtime이 담당한다.</summary>
    public class AgentChatPanel : MonoBehaviour, IAgentCommandInput
    {
        [SerializeField] CarrierAgentRuntime runtime;
        string input = "";
        Vector2 scroll;
        int shownMessageCount;
        GUIStyle wrapStyle;
        public event Action<string> CommandSubmitted;

        public void Submit(string text)
        {
            if (isActiveAndEnabled && !string.IsNullOrWhiteSpace(text))
                CommandSubmitted?.Invoke(text.Trim());
        }

        void OnGUI()
        {
            if (runtime == null || runtime.Commands == null) return;
            var client = runtime.Commands;
            wrapStyle ??= new GUIStyle(GUI.skin.label) { wordWrap = true, richText = true };
            GUILayout.BeginArea(new Rect(10f, Screen.height - 310f, 480f, 300f), GUI.skin.box);
            GUILayout.Label(client.ThreadId == null ? "Agent chat (new conversation)" : $"Agent chat · thread {client.ThreadId.Substring(0, Math.Min(8, client.ThreadId.Length))}");
            if (shownMessageCount != client.Lines.Count)
            {
                shownMessageCount = client.Lines.Count;
                scroll.y = float.MaxValue;
            }
            scroll = GUILayout.BeginScrollView(scroll, GUILayout.ExpandHeight(true));
            foreach (var line in client.Lines) GUILayout.Label(line, wrapStyle);
            GUILayout.EndScrollView();
            if (!string.IsNullOrEmpty(client.Status)) GUILayout.Label(client.Status, wrapStyle);
            if (runtime.ServerError != null) GUILayout.Label(runtime.ServerError, wrapStyle);
            bool enter = Event.current.type == EventType.KeyDown &&
                         (Event.current.keyCode == KeyCode.Return || Event.current.keyCode == KeyCode.KeypadEnter) &&
                         GUI.GetNameOfFocusedControl() == "chat_input";
            bool submit = enter;
            GUILayout.BeginHorizontal();
            GUI.SetNextControlName("chat_input");
            input = GUILayout.TextField(input, GUILayout.ExpandWidth(true));
            if (GUILayout.Button("Send", GUILayout.Width(60f))) submit = true;
            GUI.enabled = !client.IsSending;
            if (GUILayout.Button("New", GUILayout.Width(50f))) client.NewConversation();
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
            if (submit && runtime.ServerError == null && !string.IsNullOrWhiteSpace(input))
            {
                Submit(input);
                input = "";
                if (enter) Event.current.Use();
            }
        }
    }
}
