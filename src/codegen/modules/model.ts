import type { ModelConfig } from "@/components/model-builder";

export function generateModelCode(config: ModelConfig): string {
    let layers: string[] = config.customLayers?.map(layer => {
        const params = layer.config?.join(", ") || ""; 
        return `nn.${layer.type}(${params})`;
    }) || [];

    let code = [
        "# === Model Definition ===",
        "import torch.nn as nn",
        "",
        "class CustomModel(torch.nn.Module):",
        "    def __init__(self):",
        "        super(CustomModel, self).__init__()",
        `        self.layers = torch.nn.Sequential(${layers.join(", ")})`,
        "",
        "    def forward(self, x):",
        "        return self.layers(x)",
        "",
        "model = CustomModel()",
        ""
    ];
    return code.join("\n");
}