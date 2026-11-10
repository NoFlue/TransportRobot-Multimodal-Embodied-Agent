using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.AI;

namespace MultimodalAgent
{
    /// <summary>
    /// 시드로 객체 배치를 바꾼다. Terrain과 장애물은 그대로 두고, 고정 객체(화로)와 집을 수 있는 객체의 위치·방향만 정한다.
    /// 같은 시드면 같은 배치가 나온다.
    /// </summary>
    public static class WorldRandomizer
    {
        const float EdgeMargin = 4f;      // Terrain 가장자리에서 떨어질 거리
        const float MinSpacing = 4f;      // 객체끼리 최소 거리. 판정 반경(3 m)보다 커서 처음부터 '근처'인 객체가 없다.
        const float AgentClearance = 4f;  // 에이전트 시작점에서 떨어질 거리
        const int MaxTries = 400;
        const int CarriedLayer = 2;       // 집는 객체와 에이전트가 쓰는 Ignore Raycast 레이어

        /// <summary>
        /// objects를 배치한다. stationary는 옮기지 않는 객체(장애물에 붙은 label 등)의 위치로, 간격 계산에만 쓴다.
        /// 고정 객체를 옮긴 뒤 syncNavMesh로 NavMesh를 맞추고 나서 집는 객체의 경로를 검사한다.
        /// </summary>
        public static void Apply(int seed, Vector3 agentStart, IList<LabeledObject> objects, IEnumerable<Vector3> stationary, Action syncNavMesh)
        {
            var rng = new System.Random(seed);
            var terrain = Terrain.activeTerrain;
            var origin = terrain.transform.position;
            var size = terrain.terrainData.size;
            var occupied = new List<Vector3> { agentStart };
            occupied.AddRange(stationary);
            var mask = ~(1 << CarriedLayer);

            Vector3 Sample()
            {
                var x = origin.x + EdgeMargin + (float)rng.NextDouble() * (size.x - 2f * EdgeMargin);
                var z = origin.z + EdgeMargin + (float)rng.NextDouble() * (size.z - 2f * EdgeMargin);
                var point = new Vector3(x, 0f, z);
                point.y = terrain.SampleHeight(point) + origin.y;
                return point;
            }

            bool Spaced(Vector3 point, float spacing) => occupied.All(o =>
                Vector2.Distance(new Vector2(o.x, o.z), new Vector2(point.x, point.z)) >= spacing);

            bool Blocked(Vector3 center, Vector3 halfExtents, Transform self) =>
                Physics.OverlapBox(center, halfExtents, Quaternion.identity, mask, QueryTriggerInteraction.Ignore)
                    .Any(c => !(c is TerrainCollider) && (self == null || !c.transform.IsChildOf(self)));

            var fixedObjects = objects.Where(o => !o.Pickable).OrderBy(o => o.Label).ToList();
            var items = objects.Where(o => o.Pickable).OrderBy(o => o.Label).ToList();

            foreach (var obj in fixedObjects)
            {
                // 둘레를 걸어 다닐 수 있게 객체보다 넉넉한 상자로 장애물과 겹치는지 본다.
                var half = new Vector3(obj.FootprintRadius + 1.5f, 1f, obj.FootprintRadius + 1.5f);
                for (var i = 0; i < MaxTries; i++)
                {
                    var point = Sample();
                    var yaw = (float)rng.NextDouble() * 360f;
                    if (!Spaced(point, Mathf.Max(MinSpacing, AgentClearance + obj.FootprintRadius))) continue;
                    if (Blocked(point + Vector3.up * 1.1f, half, obj.transform)) continue;
                    obj.transform.SetPositionAndRotation(point, Quaternion.Euler(0f, yaw, 0f));
                    occupied.Add(point);
                    break;
                }
            }
            syncNavMesh();

            NavMesh.SamplePosition(agentStart, out var startHit, 3f, NavMesh.AllAreas);
            var path = new NavMeshPath();
            foreach (var obj in items)
            {
                for (var i = 0; i < MaxTries; i++)
                {
                    var point = Sample();
                    var yaw = (float)rng.NextDouble() * 360f;
                    if (!Spaced(point, MinSpacing)) continue;
                    if (Blocked(point + Vector3.up * 0.5f, new Vector3(0.7f, 0.4f, 0.7f), null)) continue;
                    if (!NavMesh.SamplePosition(point, out var hit, 1.0f, NavMesh.AllAreas)) continue;
                    if (!NavMesh.CalculatePath(startHit.position, hit.position, NavMesh.AllAreas, path) ||
                        path.status != NavMeshPathStatus.PathComplete) continue;
                    obj.PlaceAt(obj.transform.parent, point, Quaternion.Euler(0f, yaw, 0f));
                    occupied.Add(point);
                    break;
                }
            }
            Physics.SyncTransforms();
        }
    }
}
