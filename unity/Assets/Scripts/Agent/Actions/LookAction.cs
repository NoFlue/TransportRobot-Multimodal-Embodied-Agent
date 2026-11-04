using System.Collections;

namespace MultimodalAgent
{
    /// <summary>대상 객체를 바라보도록 몸을 돌린다.</summary>
    public sealed class LookAction : AgentAction
    {
        readonly string label;
        LabeledObject target;

        public LookAction(string label) => this.label = label;

        public override string Name => "look_at";

        public override bool Validate(CarrierAgent agent, out string error)
        {
            if (!TryFindTarget(label, out target, out error)) return false;
            if (agent.IsCarrying(target))
            {
                error = agent.CarriedTargetError(target, $"look_at {label}");
                return false;
            }
            return true;
        }

        public override IEnumerator Execute(CarrierAgent agent, ActionResult result)
        {
            yield return agent.TurnTowards(target.Position);
            result.Succeed($"Now facing {label} ({agent.DistanceTo(target):0.0} m away).");
        }
    }
}
