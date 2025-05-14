import type { ModelConfig } from "@/components/model-builder"

export interface BackendGenerators {
  dataLoader: (config: ModelConfig) => string;
  model:      (config: ModelConfig) => string;
  optimizer:  (config: ModelConfig) => string;
  metrics:    (config: ModelConfig) => string;
  training:   (config: ModelConfig) => string;
}
