using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;

namespace MultimodalAgent
{
    // Python agent/dto.py와 1:1로 맞춘 JSON 계약. 예시는 docs/contracts/*.json에 있다.

    public sealed class Vec3Dto
    {
        [JsonProperty("x")] public float X;
        [JsonProperty("y")] public float Y;
        [JsonProperty("z")] public float Z;

        public static Vec3Dto From(Vector3 v) => new Vec3Dto { X = Round(v.x), Y = Round(v.y), Z = Round(v.z) };

        static float Round(float value) => Mathf.Round(value * 100f) / 100f;
    }

    public sealed class ObjectDataDto
    {
        [JsonProperty("label")] public string Label;
        [JsonProperty("position")] public Vec3Dto Position;
        [JsonProperty("pickable")] public bool Pickable;

        public static ObjectDataDto From(ObjectData data) =>
            new ObjectDataDto { Label = data.label, Position = Vec3Dto.From(data.position), Pickable = data.pickable };
    }

    public sealed class AgentStateDto
    {
        [JsonProperty("position")] public Vec3Dto Position;
        [JsonProperty("holding")] public string Holding;
        [JsonProperty("inventory")] public List<string> Inventory = new List<string>();
    }

    public sealed class ObservationDto
    {
        [JsonProperty("timestamp")] public double Timestamp;
        [JsonProperty("agent")] public AgentStateDto Agent;
        [JsonProperty("objects")] public List<ObjectDataDto> Objects = new List<ObjectDataDto>();
        [JsonProperty("message")] public string Message;
        [JsonProperty("image_png_base64")] public string ImagePngBase64;
    }

    public sealed class ActRequestDto
    {
        [JsonProperty("action")] public string Action;
        [JsonProperty("label")] public string Label;
        [JsonProperty("near_label")] public string NearLabel;
    }

    public sealed class ActionResultDto
    {
        [JsonProperty("action")] public string Action;
        [JsonProperty("success")] public bool Success;
        [JsonProperty("feedback")] public string Feedback;
        [JsonProperty("started_at")] public double StartedAt;
        [JsonProperty("finished_at")] public double FinishedAt;
        [JsonProperty("observation")] public ObservationDto Observation;
    }

    public sealed class ViewRequestDto
    {
        /// <summary>몸의 정면 기준 고개를 돌린 각도(도). 양수는 오른쪽.</summary>
        [JsonProperty("yaw")] public float Yaw;
    }

    public sealed class LocateRequestDto
    {
        [JsonProperty("yaw")] public float Yaw;
        /// <summary>이미지 좌표 0~1. x는 왼쪽, y는 위쪽 기준.</summary>
        [JsonProperty("x")] public float X;
        [JsonProperty("y")] public float Y;
        /// <summary>객체 이름(영어). 새 label의 접두어가 된다. 예: barrel -> barrel_1</summary>
        [JsonProperty("name")] public string Name;
    }

    public sealed class LocateResultDto
    {
        [JsonProperty("found")] public bool Found;
        /// <summary>이미 등록된 객체였는지.</summary>
        [JsonProperty("known")] public bool Known;
        [JsonProperty("feedback")] public string Feedback;
        [JsonProperty("object")] public ObjectDataDto Object;
    }
}
