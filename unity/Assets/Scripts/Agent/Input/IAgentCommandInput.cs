using System;

namespace MultimodalAgent
{
    /// <summary>입력 방식과 무관한 최종 자연어 명령. Unity 메인 스레드에서 발생한다.</summary>
    public interface IAgentCommandInput
    {
        event Action<string> CommandSubmitted;
    }
}
