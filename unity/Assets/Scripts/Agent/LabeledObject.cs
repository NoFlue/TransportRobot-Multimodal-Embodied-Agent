using System.Collections.Generic;
using UnityEngine;

namespace MultimodalAgent
{
    /// <summary>
    /// 씬 안에서 label로 찾을 수 있는 객체. 관측 시점에 ObjectData를 만든다.
    /// </summary>
    public class LabeledObject : MonoBehaviour
    {
        static readonly Dictionary<string, LabeledObject> Registry = new Dictionary<string, LabeledObject>();

        [SerializeField] string label = "object_1";
        [SerializeField] bool pickable = true;

        Collider[] colliders;
        Rigidbody body;

        public string Label => label;
        public bool Pickable => pickable;
        public Vector3 Position => transform.position;

        /// <summary>바닥에서 본 객체 반경. 접근 지점과 거리 계산에 쓴다.</summary>
        public float FootprintRadius { get; private set; } = 0.25f;

        public static IEnumerable<LabeledObject> All => Registry.Values;

        public static bool TryFind(string label, out LabeledObject found)
        {
            found = null;
            return !string.IsNullOrEmpty(label) && Registry.TryGetValue(label, out found) && found != null;
        }

        public void Configure(string newLabel, bool isPickable)
        {
            label = newLabel;
            pickable = isPickable;
        }

        void Awake()
        {
            colliders = GetComponentsInChildren<Collider>(true);
            body = GetComponent<Rigidbody>();
            FootprintRadius = MeasureFootprint();
            if (Registry.TryGetValue(label, out var existing) && existing != null && existing != this)
            {
                Debug.LogError($"Duplicate object label '{label}' on {name} and {existing.name}.");
                return;
            }
            Registry[label] = this;
        }

        void OnDestroy()
        {
            if (Registry.TryGetValue(label, out var existing) && existing == this)
            {
                Registry.Remove(label);
            }
        }

        public ObjectData ToObjectData() => new ObjectData(label, transform.position, pickable);

        /// <summary>손이나 가방에 붙인다. 붙어 있는 동안 충돌과 물리를 끈다.</summary>
        public void AttachTo(Transform parent)
        {
            SetPhysicsEnabled(false);
            transform.SetParent(parent, false);
            transform.localPosition = Vector3.zero;
            transform.localRotation = Quaternion.identity;
        }

        /// <summary>월드에 내려놓는다.</summary>
        public void PlaceAt(Transform parent, Vector3 position, Quaternion rotation)
        {
            gameObject.SetActive(true);
            transform.SetParent(parent, true);
            transform.SetPositionAndRotation(position, rotation);
            SetPhysicsEnabled(true);
        }

        void SetPhysicsEnabled(bool enabled)
        {
            foreach (var c in colliders)
            {
                if (c != null) c.enabled = enabled;
            }
            if (body != null) body.isKinematic = true;
        }

        float MeasureFootprint()
        {
            var bounds = new Bounds(transform.position, Vector3.zero);
            var any = false;
            foreach (var c in colliders)
            {
                if (c == null || !c.enabled) continue;
                if (!any) { bounds = c.bounds; any = true; }
                else bounds.Encapsulate(c.bounds);
            }
            if (!any) return 0.25f;
            return Mathf.Max(bounds.extents.x, bounds.extents.z);
        }
    }
}
