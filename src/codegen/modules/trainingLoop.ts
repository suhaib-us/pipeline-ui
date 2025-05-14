import type { ModelConfig } from "@/components/model-builder"

export function generateTrainingCode(config: ModelConfig): string {
  const lines: string[] = [
    "# === Training Loop ===",
    "for epoch in range(1, " +
      (config.training?.epochs ?? 1) +
      " + 1):",
    "    model.train()",
    "    for batch in dataloader:",
    "        inputs, labels = batch",
    "        optimizer.zero_grad()",
    "        outputs = model(inputs)",
    "        loss = loss_fn(outputs, labels)",
    "        loss.backward()",
    "        optimizer.step()",
    config.training?.earlyStoppingEnabled
      ? "    # Early stopping logic would go here"
      : "",
    "print('Training complete')"
  ];
  return lines.filter(Boolean).join("\n");
}
