using System;
using UnityEngine;

namespace MultimodalAgent
{
    /// <summary>
    /// 에이전트에게 보내는 객체 정보. label로 객체를 특정하고 position으로 위치를,
    /// pickable로 집어 옮길 수 있는 객체인지(속성)를 알린다.
    /// </summary>
    [Serializable]
    public class ObjectData
    {
        public string label;
        public Vector3 position;
        public bool pickable;

        public ObjectData(string label, Vector3 position, bool pickable)
        {
            this.label = label;
            this.position = position;
            this.pickable = pickable;
        }
    }
}
