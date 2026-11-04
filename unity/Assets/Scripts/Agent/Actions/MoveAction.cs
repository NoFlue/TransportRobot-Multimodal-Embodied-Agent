using System.Collections;
using UnityEngine;

namespace MultimodalAgent
{
    /// <summary>대상 객체 옆의 도달 가능한 지점까지 이동한다.</summary>
    public sealed class MoveAction : AgentAction
    {
        const float StuckSeconds = 3f;

        readonly string label;
        LabeledObject target;

        public MoveAction(string label) => this.label = label;

        public override string Name => "move_to";

        public override bool Validate(CarrierAgent agent, out string error)
        {
            if (!TryFindTarget(label, out target, out error)) return false;
            if (agent.IsCarrying(target))
            {
                error = agent.CarriedTargetError(target, $"move_to {label}");
                return false;
            }
            return true;
        }

        public override IEnumerator Execute(CarrierAgent agent, ActionResult result)
        {
            var approach = new ApproachResult();
            yield return Approach(agent, target, approach);
            if (approach.Failed) result.Fail(approach.Message);
            else result.Succeed(approach.Message);
        }

        internal sealed class ApproachResult
        {
            public bool Failed;
            public string Message = "";
        }

        /// <summary>대상 옆까지 걸어간다. Pick과 Drop도 대상이 멀면 이 이동을 먼저 한다.</summary>
        internal static IEnumerator Approach(CarrierAgent agent, LabeledObject target, ApproachResult outcome)
        {
            var label = target.Label;
            if (agent.DistanceTo(target) <= agent.ReachDistance)
            {
                yield return agent.TurnTowards(target.Position);
                outcome.Message = $"Already next to {label} ({agent.DistanceTo(target):0.0} m).";
                yield break;
            }
            if (!agent.TryFindApproachPoint(target, out var point) || !agent.SetDestination(point))
            {
                outcome.Failed = true;
                outcome.Message = $"No reachable point next to {label}.";
                yield break;
            }

            var lastPosition = agent.Position;
            var lastProgressTime = Time.time;
            while (!agent.HasArrived)
            {
                if ((agent.Position - lastPosition).sqrMagnitude > 0.01f)
                {
                    lastPosition = agent.Position;
                    lastProgressTime = Time.time;
                }
                else if (Time.time - lastProgressTime > StuckSeconds)
                {
                    agent.StopMoving();
                    outcome.Failed = true;
                    outcome.Message = $"Movement toward {label} is blocked; stopped {agent.DistanceTo(target):0.0} m away.";
                    yield break;
                }
                yield return null;
            }
            agent.StopMoving();
            yield return agent.TurnTowards(target.Position);
            outcome.Message = $"Arrived next to {label} ({agent.DistanceTo(target):0.0} m).";
        }
    }
}
