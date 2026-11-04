using System.Collections;
using UnityEngine;

namespace MultimodalAgent
{
    /// <summary>
    /// 객체를 손에 든다. 멀면 먼저 걸어가고, 손이 차 있으면 들고 있던 객체를 인벤토리에 넣는다.
    /// </summary>
    public sealed class PickAction : AgentAction
    {
        readonly string label;
        LabeledObject target;

        public PickAction(string label) => this.label = label;

        public override string Name => "pick";

        public override bool Validate(CarrierAgent agent, out string error)
        {
            if (!TryFindTarget(label, out target, out error)) return false;
            if (!target.Pickable)
            {
                error = $"{label} cannot be picked up.";
                return false;
            }
            if (agent.IsCarrying(target))
            {
                error = $"You are already carrying {label}.";
                return false;
            }
            if (agent.Holding != null && agent.Inventory.Count >= agent.InventoryCapacity)
            {
                error = $"Your hand and inventory are full ({agent.InventoryCapacity + 1} objects). Drop something first.";
                return false;
            }
            return true;
        }

        public override IEnumerator Execute(CarrierAgent agent, ActionResult result)
        {
            if (agent.DistanceTo(target) > agent.ReachDistance)
            {
                var approach = new MoveAction.ApproachResult();
                yield return MoveAction.Approach(agent, target, approach);
                if (approach.Failed)
                {
                    result.Fail($"Could not reach {label}: {approach.Message}");
                    yield break;
                }
            }
            yield return agent.TurnTowards(target.Position);
            yield return new WaitForSeconds(0.2f);

            string stored = null;
            if (agent.Holding != null)
            {
                stored = agent.Holding.Label;
                agent.StoreHeld();
            }
            agent.TakeInHand(target);
            result.Succeed(stored == null
                ? $"Picked up {label}; it is now in your hand."
                : $"Moved {stored} from your hand to the inventory, then picked up {label}; it is now in your hand.");
        }
    }
}
