using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;
using Whisper;
using Object = UnityEngine.Object;

public class CommandInputTests
{
    GameObject target;
    Component runtime, client, chat, voice;
    const BindingFlags Fields = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    static Type AgentType(string name) => AppDomain.CurrentDomain.GetAssemblies()
        .Select(a => a.GetType("MultimodalAgent." + name)).First(t => t != null);
    static void Set(object obj, string name, object value) => obj.GetType().GetField(name, Fields).SetValue(obj, value);
    static object Get(object obj, string name) => obj.GetType().GetProperty(name, Fields).GetValue(obj);
    static object Call(object obj, string name, params object[] args) => obj.GetType().GetMethod(name, Fields).Invoke(obj, args);
    Queue Pending => new Queue((ICollection)client.GetType().GetField("pending", Fields).GetValue(client));

    [SetUp]
    public void Setup()
    {
        target = new GameObject("Input test");
        target.SetActive(false);
        client = target.AddComponent(AgentType("AgentCommandClient"));
        chat = target.AddComponent(AgentType("AgentChatPanel"));
        var manager = target.AddComponent<WhisperManager>();
        var settings = new SerializedObject(manager);
        settings.FindProperty("initOnAwake").boolValue = false;
        settings.ApplyModifiedPropertiesWithoutUndo();
        voice = target.AddComponent(AgentType("WhisperCommandInput"));
        Set(voice, "whisper", manager);
        runtime = target.AddComponent(AgentType("CarrierAgentRuntime"));
        Set(runtime, "<Commands>k__BackingField", client);
        Set(runtime, "commandInputs", new MonoBehaviour[] { (MonoBehaviour)chat, (MonoBehaviour)voice, (MonoBehaviour)chat });
        Set(client, "sending", true); // Inspect accepted commands without contacting a live Python agent.
        target.SetActive(true);
        Call(runtime, "OnDisable");
        Call(runtime, "OnEnable");
    }

    [TearDown]
    public void Cleanup()
    {
        Call(runtime, "OnDisable");
        Object.DestroyImmediate(target);
    }

    [Test]
    public void ChatRoutesThroughInterfaceOnceAndUnsubscribes()
    {
        Assert.That(AgentType("IAgentCommandInput").IsAssignableFrom(chat.GetType()), Is.True);
        Assert.That(AgentType("IAgentCommandInput").IsAssignableFrom(voice.GetType()), Is.True);
        Call(runtime, "OnEnable");
        Call(chat, "Submit", "   석탄을 가져와   ");
        Call(chat, "Submit", "   ");
        Assert.That(Get(runtime, "LastMessage"), Is.EqualTo("석탄을 가져와"));
        Assert.That(Pending.Count, Is.EqualTo(1));
        Call(runtime, "OnDisable");
        Call(chat, "Submit", "ignored");
        Assert.That(Pending.Count, Is.EqualTo(1));
        Call(runtime, "OnEnable");
        Call(chat, "Submit", "철도 가져와");
        Assert.That(Pending.ToArray(), Is.EqualTo(new[] { "석탄을 가져와", "철도 가져와" }));
    }

    [Test]
    public void VoiceRejectsSilenceShortAndDisabledInput()
    {
        ((Task)Call(voice, "TranscribeAsync", new float[16000], 16000, 1)).GetAwaiter().GetResult();
        ((Task)Call(voice, "TranscribeAsync", new float[10], 16000, 1)).GetAwaiter().GetResult();
        ((MonoBehaviour)voice).enabled = false;
        ((Task)Call(voice, "TranscribeAsync", Enumerable.Repeat(0.1f, 16000).ToArray(), 16000, 1)).GetAwaiter().GetResult();
        Assert.That(Pending.Count, Is.EqualTo(0));
    }

    [Test]
    public void RuntimeRejectsCommandsWhenActionServerFailed()
    {
        Set(runtime, "<ServerError>k__BackingField", "port conflict");
        Call(chat, "Submit", "석탄 가져와");
        Assert.That(Pending.Count, Is.EqualTo(0));
        Assert.That(Get(client, "Status"), Is.EqualTo("port conflict"));
    }

    [UnityTest]
    public IEnumerator RealWhisperTranscriptUsesSameRuntimeQueue()
    {
        Set(voice, "language", "en");
        var clip = AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/Tests/Editor/Fixtures/jfk.wav");
        Assert.That(clip, Is.Not.Null);
        clip.LoadAudioData();
        var audio = new float[clip.samples * clip.channels];
        Assert.That(clip.GetData(audio, 0), Is.True);
        Call(chat, "Submit", "first text command");
        var task = (Task)Call(voice, "TranscribeAsync", audio, clip.frequency, clip.channels);
        double deadline = EditorApplication.timeSinceStartup + 120;
        while (!task.IsCompleted && EditorApplication.timeSinceStartup < deadline) yield return null;
        Assert.That(task.IsCompleted, Is.True, "Whisper timed out");
        Assert.That(task.IsFaulted, Is.False, task.Exception?.ToString());
        Assert.That(Pending.Count, Is.EqualTo(2), Get(voice, "Status") as string);
        string transcript = (string)Get(runtime, "LastMessage");
        Assert.That(transcript.ToLowerInvariant(), Does.Contain("country"));
        Debug.Log("Whisper input integration transcript: " + transcript);
    }
}
