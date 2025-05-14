import type { ModelConfig } from "@/components/model-builder"

export function generateDataLoaderCode(config: ModelConfig): string {
  
  const filename = config.fileMetadata?.name ?? "noname.err"

  let codeLines: string[] = [
    "# === Data Loading ===",
    "import torch",
    "from torch.utils.data import DataLoader, Dataset",
    "",
  ];

  if (filename) {
    codeLines.push(
      `class CustomDataset(Dataset):`,
      `    def __init__(self, data_path):`,
      `        # load data from path`,
      `        self.data = load_data(data_path)`,
      "",
      `    def __len__(self):`,
      `        return len(self.data)`,
      "",
      `    def __getitem__(self, idx):`,
      `        return self.data[idx]`,
      "",
      `dataset = CustomDataset("${filename}")`,
      `dataloader = DataLoader(dataset, batch_size=${config.training?.batchSize}, shuffle=True)`,
      ""
    );
  }
  return codeLines.join("\n");
}