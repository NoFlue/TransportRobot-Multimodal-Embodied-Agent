using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

namespace MultimodalAgent
{
    /// <summary>
    /// 실제 움직임을 수행하는 에이전트. 이동, 회전, 집기, 보관, 내려놓기와 손·인벤토리 상태를 가진다.
    /// </summary>
    [RequireComponent(typeof(NavMeshAgent))]
    public class CarrierAgent : MonoBehaviour
    {
        [SerializeField] Camera eyeCamera;
        [SerializeField] Transform hand;
        [SerializeField] Transform backpack;
        [SerializeField] int inventoryCapacity = 4;
        [SerializeField] float reachDistance = 1.6f;
        [SerializeField] float approachMargin = 0.8f;
        [SerializeField] float turnSpeedDegrees = 270f;

        readonly List<LabeledObject> inventory = new List<LabeledObject>();
        NavMeshAgent nav;

        public Camera EyeCamera => eyeCamera;
        public LabeledObject Holding { get; private set; }
        public IReadOnlyList<LabeledObject> Inventory => inventory;
        public int InventoryCapacity => inventoryCapacity;
        /// <summary>객체 표면까지 이 거리 안이면 집거나 그 옆에 내려놓을 수 있다.</summary>
        public float ReachDistance => reachDistance;
        public Vector3 Position => transform.position;
        public bool IsMoving => nav != null && nav.enabled && nav.isOnNavMesh && (nav.pathPending || nav.hasPath);

        void Awake()
        {
            nav = GetComponent<NavMeshAgent>();
        }

        /// <summary>NavMesh가 준비된 뒤 런타임이 호출한다.</summary>
        public bool EnableNavigation()
        {
            if (!NavMesh.SamplePosition(transform.position, out var hit, 3f, NavMesh.AllAreas))
            {
                Debug.LogError("CarrierAgent start position is not on the NavMesh.");
                return false;
            }
            nav.enabled = true;
            nav.Warp(hit.position);
            return true;
        }

        public bool IsCarrying(LabeledObject obj) => obj != null && (Holding == obj || inventory.Contains(obj));

        /// <summary>들고 있는 객체를 대상으로 쓴 행동에 돌려줄 피드백. 다음에 할 행동을 알려 준다.</summary>
        public string CarriedTargetError(LabeledObject obj, string attemptedAction)
        {
            var place = Holding == obj ? "in your hand" : "in your inventory";
            return $"{obj.Label} is {place}, so you cannot {attemptedAction}; it moves with you. " +
                   $"If you need it on the ground, call drop with label {obj.Label} first.";
        }

        public static Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

        /// <summary>에이전트 중심에서 객체 표면까지의 수평 거리.</summary>
        public float DistanceTo(LabeledObject obj) =>
            Mathf.Max(0f, Flat(obj.Position - Position).magnitude - obj.FootprintRadius);

        public bool TryFindApproachPoint(LabeledObject target, out Vector3 point)
        {
            var center = target.Position;
            var radius = target.FootprintRadius + approachMargin;
            var dir = Flat(Position - center);
            dir = dir.sqrMagnitude < 1e-4f ? Vector3.forward : dir.normalized;
            var path = new NavMeshPath();
            for (var i = 0; i < 12; i++)
            {
                var step = (i + 1) / 2;
                var angle = (i % 2 == 0 ? 1f : -1f) * step * 30f;
                var candidate = center + Quaternion.Euler(0f, angle, 0f) * dir * radius;
                if (!NavMesh.SamplePosition(candidate, out var hit, 1.0f, NavMesh.AllAreas)) continue;
                if (Flat(hit.position - center).magnitude > radius + 0.6f) continue;
                if (NavMesh.CalculatePath(Position, hit.position, NavMesh.AllAreas, path) &&
                    path.status == NavMeshPathStatus.PathComplete)
                {
                    point = hit.position;
                    return true;
                }
            }
            point = default;
            return false;
        }

        public bool SetDestination(Vector3 point)
        {
            nav.isStopped = false;
            return nav.SetDestination(point);
        }

        public bool HasArrived =>
            !nav.pathPending && (!nav.hasPath || nav.remainingDistance <= nav.stoppingDistance + 0.05f);

        public void StopMoving()
        {
            if (nav != null && nav.enabled && nav.isOnNavMesh)
            {
                nav.ResetPath();
                nav.velocity = Vector3.zero;
            }
        }

        public IEnumerator TurnTowards(Vector3 worldPoint)
        {
            var dir = Flat(worldPoint - Position);
            if (dir.sqrMagnitude < 1e-4f) yield break;
            var goal = Quaternion.LookRotation(dir.normalized, Vector3.up);
            while (Quaternion.Angle(transform.rotation, goal) > 1f)
            {
                transform.rotation = Quaternion.RotateTowards(transform.rotation, goal, turnSpeedDegrees * Time.deltaTime);
                yield return null;
            }
            transform.rotation = goal;
        }

        public void TakeInHand(LabeledObject obj)
        {
            obj.AttachTo(hand);
            Holding = obj;
        }

        public void StoreHeld()
        {
            var obj = Holding;
            obj.AttachTo(backpack);
            obj.gameObject.SetActive(false);
            inventory.Add(obj);
            Holding = null;
        }

        /// <summary>손이나 인벤토리에서 꺼내 월드의 지정 위치에 놓는다.</summary>
        public void Release(LabeledObject obj, Transform worldParent, Vector3 position, Quaternion rotation)
        {
            if (Holding == obj) Holding = null;
            inventory.Remove(obj);
            obj.PlaceAt(worldParent, position, rotation);
        }

        public void ClearCarrying()
        {
            Holding = null;
            inventory.Clear();
        }

        public void Teleport(Vector3 position, Quaternion rotation)
        {
            StopMoving();
            if (nav.enabled) nav.Warp(position);
            else transform.position = position;
            transform.rotation = rotation;
        }
    }
}
