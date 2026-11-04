using System.Collections;

namespace MultimodalAgent
{
    /// <summary>행동 하나의 실행 결과. Python에 실행 피드백으로 돌아간다.</summary>
    public sealed class ActionResult
    {
        public bool Success { get; private set; }
        public string Feedback { get; private set; } = "";

        public void Succeed(string feedback)
        {
            Success = true;
            Feedback = feedback;
        }

        public void Fail(string feedback)
        {
            Success = false;
            Feedback = feedback;
        }
    }

    /// <summary>
    /// 모든 행동의 추상 클래스. Python 도구 하나가 구체 클래스 하나에 대응한다.
    /// </summary>
    public abstract class AgentAction
    {
        /// <summary>Python 도구 이름과 같다. 예: move_to</summary>
        public abstract string Name { get; }

        /// <summary>실행 전 검사. 실패하면 에이전트가 고칠 수 있도록 이유를 돌려준다.</summary>
        public abstract bool Validate(CarrierAgent agent, out string error);

        /// <summary>여러 프레임에 걸쳐 실행하고 결과를 채운다.</summary>
        public abstract IEnumerator Execute(CarrierAgent agent, ActionResult result);

        /// <summary>취소나 timeout 때 호출한다.</summary>
        public virtual void Cancel(CarrierAgent agent) => agent.StopMoving();

        protected static bool TryFindTarget(string label, out LabeledObject target, out string error)
        {
            if (LabeledObject.TryFind(label, out target))
            {
                error = null;
                return true;
            }
            error = $"Unknown label '{label}'. Use a label from the object list.";
            return false;
        }
    }
}
