using System.Collections;
using UnityEngine;

namespace MultimodalAgent
{
    /// <summary>
    /// 손이나 인벤토리의 객체를 내려놓는다. near_label이 있으면 그 객체 주변에 둔다.
    /// </summary>
    public sealed class DropAction : AgentAction
    {
        const float PlaceGap = 0.6f;
        const float MinSpacing = 0.55f;

        readonly string label;
        readonly string nearLabel;
        readonly Transform worldParent;
        LabeledObject item;
        LabeledObject anchor;

        public DropAction(string label, string nearLabel, Transform worldParent)
        {
            this.label = label;
            this.nearLabel = string.IsNullOrWhiteSpace(nearLabel) ? null : nearLabel;
            this.worldParent = worldParent;
        }

        public override string Name => "drop";

        public override bool Validate(CarrierAgent agent, out string error)
        {
            if (!TryFindTarget(label, out item, out error)) return false;
            if (!agent.IsCarrying(item))
            {
                error = $"You are not carrying {label}.";
                return false;
            }
            if (nearLabel == null) return true;
            if (!TryFindTarget(nearLabel, out anchor, out error)) return false;
            if (agent.IsCarrying(anchor))
            {
                error = $"{nearLabel} is {(agent.Holding == anchor ? "in your hand" : "in your inventory")}, so nothing can be placed next to it. " +
                        $"First call drop with label {nearLabel} (no near_label) to put it on the ground, then drop {label} with near_label {nearLabel}.";
                return false;
            }
            return true;
        }

        public override IEnumerator Execute(CarrierAgent agent, ActionResult result)
        {
            if (anchor != null && agent.DistanceTo(anchor) > agent.ReachDistance + 0.5f)
            {
                // 기준 객체가 멀면 먼저 그 옆으로 걸어간다.
                var approach = new MoveAction.ApproachResult();
                yield return MoveAction.Approach(agent, anchor, approach);
                if (approach.Failed)
                {
                    result.Fail($"Could not reach {nearLabel}: {approach.Message}");
                    yield break;
                }
            }
            if (anchor != null) yield return agent.TurnTowards(anchor.Position);
            yield return new WaitForSeconds(0.2f);

            var point = anchor != null ? FindSpotNear(agent, anchor) : agent.Position + agent.transform.forward * 1.0f;
            point.y = GroundHeight(point);
            agent.Release(item, worldParent, point, Quaternion.Euler(0f, Random.Range(0f, 360f), 0f));

            result.Succeed(anchor != null
                ? $"Dropped {label} next to {nearLabel} ({CarrierAgent.Flat(item.Position - anchor.Position).magnitude:0.0} m from its center)."
                : $"Dropped {label} on the ground in front of you.");
        }

        Vector3 FindSpotNear(CarrierAgent agent, LabeledObject target)
        {
            var center = target.Position;
            var dir = CarrierAgent.Flat(agent.Position - center);
            dir = dir.sqrMagnitude < 1e-4f ? Vector3.forward : dir.normalized;
            var radius = target.FootprintRadius + PlaceGap;
            for (var i = 0; i < 12; i++)
            {
                var step = (i + 1) / 2;
                var angle = (i % 2 == 0 ? 1f : -1f) * step * 25f;
                var candidate = center + Quaternion.Euler(0f, angle, 0f) * dir * radius;
                if (IsFree(candidate)) return candidate;
            }
            return center + dir * radius;
        }

        bool IsFree(Vector3 point)
        {
            foreach (var other in LabeledObject.All)
            {
                if (other == item || !other.gameObject.activeInHierarchy || other.Pickable == false) continue;
                if (CarrierAgent.Flat(other.Position - point).magnitude < MinSpacing) return false;
            }
            return true;
        }

        static float GroundHeight(Vector3 point)
        {
            var origin = new Vector3(point.x, point.y + 20f, point.z);
            if (Physics.Raycast(origin, Vector3.down, out var hit, 50f, 1 << 0, QueryTriggerInteraction.Ignore))
            {
                return hit.point.y;
            }
            var terrain = Terrain.activeTerrain;
            return terrain != null ? terrain.SampleHeight(point) + terrain.transform.position.y : point.y;
        }
    }
}
