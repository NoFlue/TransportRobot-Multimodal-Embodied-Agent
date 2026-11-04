using System.Collections;
using UnityEngine;

namespace MultimodalAgent
{
    /// <summary>손에 든 객체를 인벤토리에 넣어 손을 비운다.</summary>
    public sealed class StoreAction : AgentAction
    {
        public override string Name => "store";

        public override bool Validate(CarrierAgent agent, out string error)
        {
            if (agent.Holding == null)
            {
                error = "Your hand is empty; there is nothing to store.";
                return false;
            }
            if (agent.Inventory.Count >= agent.InventoryCapacity)
            {
                error = $"The inventory is full ({agent.InventoryCapacity} items).";
                return false;
            }
            error = null;
            return true;
        }

        public override IEnumerator Execute(CarrierAgent agent, ActionResult result)
        {
            var label = agent.Holding.Label;
            yield return new WaitForSeconds(0.2f);
            agent.StoreHeld();
            result.Succeed($"Stored {label} in the inventory ({agent.Inventory.Count}/{agent.InventoryCapacity}); your hand is empty.");
        }
    }
}
