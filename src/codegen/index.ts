import type { ModelConfig } from "@/components/model-builder"
import { PyTorchGenerators } from "./backends/pytorch"

export function generatePythonCode(config: ModelConfig): string {
  const sections = [
    PyTorchGenerators.dataLoader(config),
    PyTorchGenerators.model(config),
    PyTorchGenerators.optimizer(config),
    PyTorchGenerators.metrics(config),
    PyTorchGenerators.training(config),
  ]
  return sections.filter(Boolean).join("\n\n")
}
