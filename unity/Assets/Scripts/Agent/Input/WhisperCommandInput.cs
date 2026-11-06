using System;
using System.Threading.Tasks;
using UnityEngine;
using Whisper;

namespace MultimodalAgent
{
    /// <summary>버튼으로 녹음한 음성을 로컬 Whisper로 변환하는 입력 소스.</summary>
    [RequireComponent(typeof(WhisperManager))]
    public class WhisperCommandInput : MonoBehaviour, IAgentCommandInput
    {
        [SerializeField] WhisperManager whisper;
        [SerializeField] string language = "ko";
        [SerializeField] string microphoneDevice = "";
        [SerializeField, Range(1, 60)] int maxRecordingSeconds = 30;
        [SerializeField, Min(0)] float silenceRmsThreshold = 0.002f;
        AudioClip recording;
        string activeDevice;
        float recordingStarted;
        int lastPosition;
        int generation;
        bool transcribing;
        bool sawSamples;

        public event Action<string> CommandSubmitted;
        public bool IsRecording => recording != null;
        public bool IsTranscribing => transcribing;
        public string Status { get; private set; } = "음성 입력 준비";

        void Awake()
        {
            if (whisper == null) whisper = GetComponent<WhisperManager>();
        }

        public void StartRecording()
        {
            if (!isActiveAndEnabled || IsRecording || transcribing) return;
            try
            {
                if (Microphone.devices.Length == 0)
                {
                    Status = "마이크가 없습니다. 장치와 마이크 권한을 확인해 주세요.";
                    return;
                }
                activeDevice = string.IsNullOrWhiteSpace(microphoneDevice) ? Microphone.devices[0] : microphoneDevice;
                if (Array.IndexOf(Microphone.devices, activeDevice) < 0)
                {
                    Status = "설정된 마이크를 찾을 수 없습니다.";
                    return;
                }
                Microphone.GetDeviceCaps(activeDevice, out int min, out int max);
                int rate = max > 0 ? Mathf.Clamp(16000, min, max) : 16000;
                recording = Microphone.Start(activeDevice, false, maxRecordingSeconds, rate);
                recordingStarted = Time.realtimeSinceStartup;
                lastPosition = 0;
                sawSamples = false;
                Status = recording != null ? "녹음 중 — 완료하려면 Stop & Send" : "마이크를 시작할 수 없습니다.";
            }
            catch (Exception e)
            {
                ReleaseRecording();
                Status = $"마이크 오류: {e.Message}";
            }
        }

        void Update()
        {
            if (!IsRecording) return;
            int position = Microphone.GetPosition(activeDevice);
            if (position > 0) { lastPosition = position; sawSamples = true; }
            float elapsed = Time.realtimeSinceStartup - recordingStarted;
            if (!sawSamples && elapsed > 3f)
            {
                ReleaseRecording();
                Status = "마이크 입력이 없습니다. Windows 마이크 권한을 확인해 주세요.";
            }
            else if (sawSamples && (!Microphone.IsRecording(activeDevice) || elapsed >= maxRecordingSeconds - 0.1f))
                StopAndSend();
        }

        public async void StopAndSend()
        {
            if (!IsRecording) return;
            try
            {
                int position = Microphone.GetPosition(activeDevice);
                int frames = position > 0 ? position : lastPosition;
                int rate = recording.frequency;
                int channels = recording.channels;
                var samples = new float[frames * channels];
                if (samples.Length > 0) recording.GetData(samples, 0);
                ReleaseRecording();
                await TranscribeAsync(samples, rate, channels);
            }
            catch (Exception e)
            {
                ReleaseRecording();
                Status = $"음성 입력 오류: {e.Message}";
            }
        }

        /// <summary>최종 오디오만 인식하여 한 번 전달한다. 비활성화된 입력의 늦은 결과는 폐기한다.</summary>
        public async Task TranscribeAsync(float[] samples, int frequency, int channels)
        {
            if (!isActiveAndEnabled || transcribing) return;
            if (samples == null || frequency <= 0 || channels <= 0 || samples.Length < frequency * channels / 4)
            {
                Status = "녹음이 너무 짧습니다. 다시 말해 주세요.";
                return;
            }
            double energy = 0;
            foreach (float sample in samples) energy += sample * sample;
            if (Math.Sqrt(energy / samples.Length) < silenceRmsThreshold)
            {
                Status = "소리가 감지되지 않았습니다. 명령을 보내지 않았습니다.";
                return;
            }
            int currentGeneration = generation;
            transcribing = true;
            try
            {
                Status = "Whisper 음성 인식 중…";
                whisper.language = language;
                whisper.translateToEnglish = false;
                if (!whisper.IsLoaded && !whisper.IsLoading) await whisper.InitModel();
                if (this == null || !isActiveAndEnabled || currentGeneration != generation) return;
                var result = await whisper.GetTextAsync(samples, frequency, channels);
                if (this == null || !isActiveAndEnabled || currentGeneration != generation) return;
                string text = result?.Result?.Trim();
                if (string.IsNullOrWhiteSpace(text))
                {
                    Status = "인식 결과가 없습니다. 모델과 마이크를 확인해 주세요.";
                    return;
                }
                Status = $"인식: {text}";
                CommandSubmitted?.Invoke(text);
            }
            catch (Exception e)
            {
                if (this != null && currentGeneration == generation) Status = $"Whisper 오류: {e.Message}";
            }
            finally { transcribing = false; }
        }

        void ReleaseRecording()
        {
            if (recording == null) return;
            Microphone.End(activeDevice);
            Destroy(recording);
            recording = null;
        }

        void OnDisable()
        {
            generation++;
            ReleaseRecording();
            Status = "음성 입력 준비";
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10f, Screen.height - 405f, 480f, 90f), GUI.skin.box);
            GUILayout.Label(Status, new GUIStyle(GUI.skin.label) { wordWrap = true });
            GUI.enabled = !transcribing;
            if (GUILayout.Button(IsRecording ? "Stop & Send" : "Voice (Korean)"))
            {
                if (IsRecording) StopAndSend(); else StartRecording();
            }
            GUI.enabled = true;
            GUILayout.EndArea();
        }
    }
}
