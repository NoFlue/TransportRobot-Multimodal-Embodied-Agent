using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;

namespace MultimodalAgent
{
    /// <summary>
    /// 사전에 등록되지 않은 객체를 관측 이미지 좌표로 특정해 LabeledObject로 등록한다.
    /// VLM이 짚은 점 주변에 광선을 여러 개 쏘고, 지면이 아닌 물체 중 가장 많이 맞은 것을 고른다.
    /// (작은 VLM은 y 좌표를 물체보다 조금 아래로 짚는 경향이 있어 위쪽으로 더 넓게 쏜다.)
    /// </summary>
    public sealed class ObjectLocator
    {
        const int SceneMask = 1 << 0; // Default 레이어: 지형과 장애물. 에이전트와 운반물은 Ignore Raycast 레이어다.
        const float MaxPickableSize = 0.8f;

        readonly List<(LabeledObject obj, Transform parent, Vector3 position, Quaternion rotation)> discovered =
            new List<(LabeledObject, Transform, Vector3, Quaternion)>();

        public LocateResultDto Locate(Camera eye, LocateRequestDto request)
        {
            var votes = new Dictionary<Collider, int>();
            var groundOnly = false;
            CarrierAgentRuntime.WithYaw(eye, request.Yaw, () =>
            {
                for (var iy = -2; iy <= 8; iy++)
                {
                    for (var ix = -1; ix <= 1; ix++)
                    {
                        var viewport = new Vector3(
                            Mathf.Clamp01(request.X + ix * 0.02f),
                            Mathf.Clamp01(1f - request.Y + iy * 0.02f),
                            0f);
                        var ray = eye.ViewportPointToRay(viewport);
                        if (!Physics.Raycast(ray, out var hit, eye.farClipPlane, SceneMask, QueryTriggerInteraction.Ignore))
                        {
                            continue;
                        }
                        if (hit.collider is TerrainCollider)
                        {
                            groundOnly = true;
                            continue;
                        }
                        votes.TryGetValue(hit.collider, out var count);
                        votes[hit.collider] = count + 1;
                    }
                }
            });

            if (votes.Count == 0)
            {
                return new LocateResultDto
                {
                    Found = false,
                    Feedback = groundOnly
                        ? "Only the ground is at that point of the view."
                        : "Nothing is at that point of the view.",
                };
            }

            var collider = votes.OrderByDescending(v => v.Value).First().Key;
            var root = collider.attachedRigidbody != null ? collider.attachedRigidbody.transform : collider.transform;
            var existing = root.GetComponentInParent<LabeledObject>();
            if (existing != null)
            {
                return new LocateResultDto
                {
                    Found = true,
                    Known = true,
                    Feedback = $"That is the already known object {existing.Label}.",
                    Object = ObjectDataDto.From(existing.ToObjectData()),
                };
            }

            var label = NextLabel(Sanitize(request.Name));
            var pickable = MeasureSize(root) <= MaxPickableSize;
            var parent = root.parent;
            var position = root.position;
            var rotation = root.rotation;
            var labeled = LabeledObject.Register(root.gameObject, label, pickable);
            discovered.Add((labeled, parent, position, rotation));
            Debug.Log($"ObjectLocator registered {label} on {root.name} (pickable={pickable}).");
            return new LocateResultDto
            {
                Found = true,
                Known = false,
                Feedback = $"Registered {root.name} as {label}.",
                Object = ObjectDataDto.From(labeled.ToObjectData()),
            };
        }

        /// <summary>월드 초기화 때 발견한 객체를 원래 자리로 돌리고 등록을 지운다.</summary>
        public void Forget()
        {
            foreach (var (obj, parent, position, rotation) in discovered)
            {
                if (obj == null) continue;
                obj.PlaceAt(parent, position, rotation);
                UnityEngine.Object.Destroy(obj);
            }
            discovered.Clear();
        }

        static string Sanitize(string name)
        {
            var text = Regex.Replace((name ?? "").Trim().ToLowerInvariant(), "[^a-z0-9]+", "_").Trim('_');
            text = Regex.Replace(text, "_[0-9]+$", "");
            return text.Length == 0 ? "object" : text;
        }

        static string NextLabel(string stem)
        {
            var index = 1;
            while (LabeledObject.TryFind($"{stem}_{index}", out _)) index++;
            return $"{stem}_{index}";
        }

        static float MeasureSize(Transform root)
        {
            var colliders = root.GetComponentsInChildren<Collider>();
            if (colliders.Length == 0) return float.MaxValue;
            var bounds = colliders[0].bounds;
            foreach (var c in colliders) bounds.Encapsulate(c.bounds);
            return Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
        }
    }
}
