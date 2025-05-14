import type { ModelConfig } from "@/components/model-builder"

export function generateMetricsCode(config: ModelConfig): string {
  const lines: string[] = [
    "# === Metrics ==="
  ];
  if (config.metrics) {
    const { name } = config.metrics;
    lines.push(`metric = torchmetrics.${name}()`);
  } else {
    lines.push("# No metrics configured");
  }
  return lines.join("\n");
}
