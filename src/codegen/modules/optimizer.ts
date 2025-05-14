import type { ModelConfig } from "@/components/model-builder"

export function generateOptimizerCode(config: ModelConfig): string {
  const lines: string[] = [
    "# === Optimizer ==="
  ];
  if (config.optimizer) {
    const { name, params } = config.optimizer;
    const paramsStr = params && Object.keys(params).length
      ? JSON.stringify(params)
      : "{}";
    lines.push(
      `optimizer = torch.optim.${name}(model.parameters(), **${paramsStr})`
    );
  } else {
    lines.push("# No optimizer configured");
  }
  return lines.join("\n");
}
