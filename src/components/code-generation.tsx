"use client"

import { useEffect, useState } from "react"
import type { ModelConfig } from "@/components/model-builder"
import { generatePythonCode } from "@/codegen"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Check, Copy, Download, Code } from "lucide-react"
import { motion } from "framer-motion"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface CodeGenerationProps {
  config: ModelConfig
}

export function CodeGeneration({ config }: CodeGenerationProps) {
  const [pythonCode, setPythonCode] = useState("")
  const [copied, setCopied] = useState(false)

  useEffect(() => {
  try {
    console.log("Config passed to codegen:", config)
    const code = generatePythonCode(config)
    if (typeof code === "string") {
      setPythonCode(code)
    } else {
      setPythonCode("# Code generation failed: output not a string")
    }
  } catch (e) {
    console.error("Code generation error:", e)
    setPythonCode("# Code generation error: " + String(e))
  }
  }, [config])


  const copyToClipboard = () => {
    navigator.clipboard.writeText(pythonCode)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const downloadCode = () => {
    const blob = new Blob([pythonCode], { type: "text/plain" })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement("a")
    a.href     = url
    a.download = "pytorch_model.py"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h2 className="text-2xl font-bold">Code Generation</h2>
        <p className="text-slate-500">Review and download generated PyTorch code</p>
      </motion.div>

      <Tabs defaultValue="code" className="w-full">
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="code"><Code className="mr-2 h-4 w-4" />Generated Code</TabsTrigger>
          <TabsTrigger value="summary">Configuration Summary</TabsTrigger>
        </TabsList>

        <TabsContent value="code" className="mt-4">
          <Card className="shadow-sm hover:shadow-md">
            <CardHeader className="flex justify-between items-center">
              <div>
                <CardTitle>PyTorch Model Code</CardTitle>
                <CardDescription>Ready-to-run script</CardDescription>
              </div>
              <div className="flex space-x-2">
                <Button variant="outline" size="sm" onClick={copyToClipboard}
                  className={copied ? "bg-green-50 text-green-600" : "hover:bg-slate-100"}>
                  {copied
                    ? <><Check className="mr-2 h-4 w-4"/><span>Copied</span></>
                    : <><Copy className="mr-2 h-4 w-4"/><span>Copy Code</span></>
                  }
                </Button>
                <Button variant="default" size="sm" onClick={downloadCode}>
                  <Download className="mr-2 h-4 w-4"/><span>Download</span>
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[500px] border rounded-md">
                <pre className="p-4 text-sm whitespace-pre-wrap">{pythonCode}</pre>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="summary" className="mt-4">
          {/* existing summary UI unchanged */}
        </TabsContent>
      </Tabs>
    </div>
  )
}



// "use client"

// import { useEffect, useState } from "react"
// import type { ModelConfig } from "./model-builder"
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
// import { ScrollArea } from "@/components/ui/scroll-area"
// import { Button } from "@/components/ui/button"
// import { Check, Copy, Download, Code } from "lucide-react"
// import { motion } from "framer-motion"
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// interface CodeGenerationProps {
//   config: ModelConfig
// }

// export function CodeGeneration({ config }: CodeGenerationProps) {
//   const [pythonCode, setPythonCode] = useState("")
//   const [copied, setCopied] = useState(false)

//   useEffect(() => {
//     // Generate Python code based on the configuration
//     const code = generatePythonCode(config)
//     setPythonCode(code)
//   }, [config])

//   const copyToClipboard = () => {
//     navigator.clipboard.writeText(pythonCode)
//     setCopied(true)
//     setTimeout(() => setCopied(false), 2000)
//   }

//   const downloadCode = () => {
//     const element = document.createElement("a")
//     const file = new Blob([pythonCode], { type: "text/plain" })
//     element.href = URL.createObjectURL(file)
//     element.download = "pytorch_model.py"
//     document.body.appendChild(element)
//     element.click()
//     document.body.removeChild(element)
//   }

//   return (
//     <div className="space-y-6">
//       <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
//         <h2 className="text-2xl font-bold text-slate-900">Code Generation</h2>
//         <p className="text-slate-500">Review your configuration and download the generated PyTorch code</p>
//       </motion.div>

//       <Tabs defaultValue="code" className="w-full">
//         <motion.div
//           initial={{ opacity: 0, y: 10 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 0.3, delay: 0.1 }}
//         >
//           <TabsList className="grid w-full grid-cols-2">
//             <TabsTrigger value="code" className="transition-all duration-200 data-[state=active]:shadow-md">
//               <Code className="mr-2 h-4 w-4" />
//               Generated Code
//             </TabsTrigger>
//             <TabsTrigger value="summary" className="transition-all duration-200 data-[state=active]:shadow-md">
//               Configuration Summary
//             </TabsTrigger>
//           </TabsList>
//         </motion.div>

//         <TabsContent value="code" className="mt-4">
//           <motion.div
//             initial={{ opacity: 0, y: 20 }}
//             animate={{ opacity: 1, y: 0 }}
//             transition={{ duration: 0.4, delay: 0.2 }}
//             whileHover={{ y: -2 }}
//           >
//             <Card className="border-slate-200 shadow-sm transition-all duration-300 hover:shadow-md">
//               <CardHeader className="flex flex-row items-center justify-between">
//                 <div>
//                   <CardTitle>PyTorch Model Code</CardTitle>
//                   <CardDescription>Ready-to-run Python code for your model</CardDescription>
//                 </div>
//                 <div className="flex space-x-2">
//                   <Button
//                     variant="outline"
//                     size="sm"
//                     onClick={copyToClipboard}
//                     className={`transition-all duration-200 ${
//                       copied ? "bg-green-50 text-green-600" : "hover:bg-slate-100"
//                     }`}
//                   >
//                     {copied ? (
//                       <motion.div
//                         className="flex items-center"
//                         initial={{ scale: 0.8 }}
//                         animate={{ scale: 1 }}
//                         transition={{ type: "spring", stiffness: 500, damping: 15 }}
//                       >
//                         <Check className="mr-2 h-4 w-4" />
//                         Copied
//                       </motion.div>
//                     ) : (
//                       <motion.div className="flex items-center">
//                         <Copy className="mr-2 h-4 w-4" />
//                         Copy Code
//                       </motion.div>
//                     )}
//                   </Button>
//                   <Button variant="default" size="sm" onClick={downloadCode} className="transition-all duration-200">
//                     <Download className="mr-2 h-4 w-4" />
//                     Download
//                   </Button>
//                 </div>
//               </CardHeader>
//               <CardContent>
//                 <ScrollArea className="h-[500px] rounded-md border">
//                   <pre className="p-4 text-sm">{pythonCode}</pre>
//                 </ScrollArea>
//               </CardContent>
//             </Card>
//           </motion.div>
//         </TabsContent>

//         <TabsContent value="summary" className="mt-4">
//           <motion.div
//             initial={{ opacity: 0, y: 20 }}
//             animate={{ opacity: 1, y: 0 }}
//             transition={{ duration: 0.4, delay: 0.2 }}
//           >
//             <Card className="border-slate-200 shadow-sm">
//               <CardHeader>
//                 <CardTitle>Configuration Summary</CardTitle>
//                 <CardDescription>Overview of your model configuration</CardDescription>
//               </CardHeader>
//               <CardContent>
//                 <ScrollArea className="h-[500px]">
//                   <div className="space-y-6">
//                     <ConfigSection
//                       title="Framework"
//                       items={[{ label: "Mode", value: config.mode === "dl" ? "Deep Learning" : "Machine Learning" }]}
//                     />

//                     <ConfigSection
//                       title="Task"
//                       items={[
//                         { label: "Main Task", value: config.mainTask },
//                         { label: "Sub-task", value: config.subTask },
//                       ]}
//                     />

//                     <ConfigSection
//                       title="Data"
//                       items={[
//                         { label: "Main Data Type", value: config.mainDataType },
//                         { label: "Data Format", value: config.subDataType },
//                       ]}
//                     />

//                     {config.fileMetadata && (
//                       <ConfigSection
//                         title="Uploaded File"
//                         items={[
//                           { label: "File Name", value: config.fileMetadata.name },
//                           { label: "File Size", value: config.fileMetadata.size.toString() },
//                           { label: "File Type", value: config.fileMetadata.type },
//                           { label: "Last Modified", value: new Date(config.fileMetadata.lastModified).toLocaleString() },
//                         ]}
//                       />
//                     )}

//                     {config.preprocessing && config.preprocessing.length > 0 && (
//                       <ConfigSection
//                         title="Preprocessing"
//                         items={config.preprocessing.map((p) => ({ label: "Transform", value: p }))}
//                       />
//                     )}

//                     <ConfigSection
//                       title="Model"
//                       items={[
//                         {
//                           label: "Model Type",
//                           value: config.modelType === "pretrained" ? "Pre-trained Model" : "Custom Model",
//                         },
//                         ...(config.modelType === "pretrained" && config.pretrainedModel
//                           ? [{ label: "Selected Model", value: config.pretrainedModel }]
//                           : []),
//                         ...(config.modelType === "custom" && config.customLayers && config.customLayers.length > 0
//                           ? [{ label: "Custom Layers", value: `${config.customLayers.length} layers defined` }]
//                           : []),
//                       ]}
//                     />

//                     {config.monitoring && (
//                       <ConfigSection
//                         title="Monitoring"
//                         items={[
//                           { label: "Category", value: config.monitoring.category },
//                           { label: "Option", value: config.monitoring.option },
//                         ]}
//                       />
//                     )}

//                     {config.optimizer && (
//                       <ConfigSection
//                         title="Optimizer"
//                         items={[
//                           { label: "Category", value: config.optimizer.category },
//                           { label: "Optimizer", value: config.optimizer.name },
//                         ]}
//                       />
//                     )}

//                     {config.loss && (
//                       <ConfigSection
//                         title="Loss Function"
//                         items={[
//                           { label: "Category", value: config.loss.category },
//                           { label: "Loss", value: config.loss.name },
//                         ]}
//                       />
//                     )}

//                     {config.metrics && (
//                       <ConfigSection
//                         title="Evaluation Metric"
//                         items={[
//                           { label: "Category", value: config.metrics.category },
//                           { label: "Metric", value: config.metrics.name },
//                         ]}
//                       />
//                     )}

//                     {config.training && (
//                       <ConfigSection
//                         title="Training Configuration"
//                         items={[
//                           { label: "Batch Size", value: config.training.batchSize.toString() },
//                           { label: "Epochs", value: config.training.epochs.toString() },
//                           { label: "Learning Rate", value: config.training.learningRate.toString() },
//                           { label: "Weight Decay", value: config.training.weightDecay.toString() },
//                           {
//                             label: "Early Stopping",
//                             value: config.training.earlyStoppingEnabled ? "Enabled" : "Disabled",
//                           },
//                           ...(config.training.earlyStoppingEnabled
//                             ? [
//                                 { label: "Patience", value: config.training.earlyStoppingPatience?.toString() || "5" },
//                                 {
//                                   label: "Min Delta",
//                                   value: config.training.earlyStoppingMinDelta?.toString() || "0.0001",
//                                 },
//                               ]
//                             : []),
//                           { label: "Scheduler", value: config.training.scheduler || "None" },
//                         ]}
//                       />
//                     )}
//                   </div>
//                 </ScrollArea>
//               </CardContent>
//             </Card>
//           </motion.div>
//         </TabsContent>
//       </Tabs>
//     </div>
//   )
// }

// interface ConfigSectionProps {
//   title: string
//   items: Array<{ label: string; value?: string }>
// }

// function ConfigSection({ title, items }: ConfigSectionProps) {
//   return (
//     <div className="rounded-lg border border-slate-200 overflow-hidden">
//       <div className="bg-slate-50 px-4 py-2 border-b border-slate-200">
//         <h3 className="font-medium text-slate-800">{title}</h3>
//       </div>
//       <div className="p-4">
//         <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
//           {items.map((item, index) => (
//             <div key={index} className="flex flex-col">
//               <dt className="text-sm font-medium text-slate-500">{item.label}</dt>
//               <dd className="text-sm text-slate-900">{item.value || "Not specified"}</dd>
//             </div>
//           ))}
//         </dl>
//       </div>
//     </div>
//   )
// }

// function generatePythonCode(config: ModelConfig): string {
//   let fileMetadataComment = "";
//   if (config.fileMetadata) {
//     fileMetadataComment = `# File Name: ${config.fileMetadata.name}\n` +
//                           `# File Size: ${config.fileMetadata.size}\n` +
//                           `# File Type: ${config.fileMetadata.type}\n` +
//                           `# Last Modified: ${new Date(config.fileMetadata.lastModified).toLocaleString()}\n\n`;
//   }

//   // Generate imports
//   let imports = fileMetadataComment + `# PyTorch Model: ${config.subTask}
// # Generated by PyTorch Model Builder

// import torch
// import torch.nn as nn
// import torch.optim as optim
// from torch.utils.data import DataLoader, Dataset
// `;

//   // Add specific imports based on configuration
//   if (config.mainDataType === "Image Data") {
//     imports += `import torchvision
// from torchvision import transforms
// from PIL import Image
// `
//   } else if (config.mainDataType === "Text Data") {
//     imports += `import torchtext
// from torchtext.data.utils import get_tokenizer
// `
//   } else if (config.mainDataType === "Audio Data") {
//     imports += `import torchaudio
// from torchaudio import transforms as audio_transforms
// `
//   }

//   if (config.metrics && config.metrics.name && config.metrics.name.startsWith("torchmetrics")) {
//     imports += `import torchmetrics
// `
//   }

//   // Add numpy and other common imports
//   imports += `import numpy as np
// import os
// import matplotlib.pyplot as plt
// from tqdm import tqdm
// `

//   // Data loading and preprocessing
//   let dataLoading = `
// # Data Loading and Preprocessing
// class CustomDataset(Dataset):
//     def __init__(self, data_path, transform=None):
//         self.data_path = data_path
//         self.transform = transform
//         # TODO: Implement dataset loading logic
        
//     def __len__(self):
//         # TODO: Return the size of the dataset
//         return 100  # Placeholder
        
//     def __getitem__(self, idx):
//         # TODO: Return the item at index idx
//         # Placeholder implementation
//         if torch.is_tensor(idx):
//             idx = idx.tolist()
            
//         # Create dummy data based on the selected data type
//         if "${config.mainDataType}" == "Image Data":
//             # Create a dummy image tensor
//             sample = torch.randn(3, 224, 224)
//         elif "${config.mainDataType}" == "Text Data":
//             # Create a dummy text tensor
//             sample = torch.randint(0, 1000, (100,))
//         elif "${config.mainDataType}" == "Audio Data":
//             # Create a dummy audio tensor
//             sample = torch.randn(1, 16000)
//         else:
//             # Create a dummy feature tensor
//             sample = torch.randn(10)
            
//         # Create a dummy label
//         label = torch.randint(0, 10, (1,)).item()
        
//         if self.transform:
//             sample = self.transform(sample)
            
//         return sample, label

// # Define transformations based on data type
// def get_transforms():
// `

//   // Add preprocessing based on data type
//   if (config.mainDataType === "Image Data") {
//     dataLoading += `    transform = transforms.Compose([
// `
//     if (config.preprocessing && config.preprocessing.length > 0) {
//       config.preprocessing.forEach((p) => {
//         dataLoading += `        ${p},\n`
//       })
//     } else {
//       dataLoading += `        transforms.Resize((224, 224)),
//         transforms.ToTensor(),
//         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
// `
//     }
//     dataLoading += `    ])
//     return transform
// `
//   } else if (config.mainDataType === "Text Data") {
//     dataLoading += `    # Text transformations
//     # TODO: Implement text-specific transformations
//     return None
// `
//   } else if (config.mainDataType === "Audio Data") {
//     dataLoading += `    # Audio transformations
//     transform = nn.Sequential(
// `
//     if (config.preprocessing && config.preprocessing.length > 0) {
//       config.preprocessing.forEach((p) => {
//         dataLoading += `        ${p},\n`
//       })
//     } else {
//       dataLoading += `        audio_transforms.MelSpectrogram(sample_rate=16000, n_mels=64),
//         audio_transforms.AmplitudeToDB(),
// `
//     }
//     dataLoading += `    )
//     return transform
// `
//   } else {
//     dataLoading += `    # Generic transformations
//     # TODO: Implement data-specific transformations
//     return None
// `
//   }

//   dataLoading += `
// # Create data loaders
// def create_data_loaders(data_path, batch_size=${config.training?.batchSize || 32}, train_ratio=0.8):
//     transform = get_transforms()
    
//     # Create a dataset
//     dataset = CustomDataset(data_path, transform=transform)
    
//     # Split into train and validation sets
//     train_size = int(train_ratio * len(dataset))
//     val_size = len(dataset) - train_size
//     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
//     # Create data loaders
//     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
//     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
//     return train_loader, val_loader
// `

//   // Model definition
//   let modelDefinition = `
// # Model Definition
// `

//   if (config.modelType === "pretrained") {
//     // Pre-trained model
//     if (config.pretrainedModel) {
//       if (config.mainTask === "Image Processing") {
//         modelDefinition += `class ${getModelClassName(config)}(nn.Module):
//     def __init__(self, num_classes=10):
//         super(${getModelClassName(config)}, self).__init__()
//         # Load pre-trained model
//         self.model = torchvision.models.${getPretrainedModelCode(config.pretrainedModel)}
        
//         # Modify the final layer for our specific task
//         if "${config.subTask}" == "Image Classification":
//             in_features = self.model.fc.in_features
//             self.model.fc = nn.Linear(in_features, num_classes)
//         elif "${config.subTask}" == "Object Detection":
//             # TODO: Implement object detection specific modifications
//             pass
//         elif "${config.subTask}" == "Image Segmentation":
//             # TODO: Implement segmentation specific modifications
//             pass
        
//     def forward(self, x):
//         return self.model(x)
// `
//       } else if (config.mainTask === "Text Processing") {
//         modelDefinition += `class ${getModelClassName(config)}(nn.Module):
//     def __init__(self, num_classes=10):
//         super(${getModelClassName(config)}, self).__init__()
//         # TODO: Load pre-trained NLP model
//         # This is a placeholder implementation
//         self.embedding = nn.Embedding(10000, 300)
//         self.encoder = nn.LSTM(300, 512, batch_first=True)
//         self.fc = nn.Linear(512, num_classes)
        
//     def forward(self, x):
//         x = self.embedding(x)
//         _, (hidden, _) = self.encoder(x)
//         return self.fc(hidden[-1])
// `
//       } else {
//         modelDefinition += `class ${getModelClassName(config)}(nn.Module):
//     def __init__(self, num_classes=10):
//         super(${getModelClassName(config)}, self).__init__()
//         # TODO: Implement model for ${config.subTask}
//         # This is a placeholder implementation
//         self.layers = nn.Sequential(
//             nn.Linear(10, 128),
//             nn.ReLU(),
//             nn.Linear(128, 64),
//             nn.ReLU(),
//             nn.Linear(64, num_classes)
//         )
        
//     def forward(self, x):
//         return self.layers(x)
// `
//       }
//     } else {
//       modelDefinition += `class ${getModelClassName(config)}(nn.Module):
//     def __init__(self, num_classes=10):
//         super(${getModelClassName(config)}, self).__init__()
//         # TODO: Implement model for ${config.subTask}
//         # This is a placeholder implementation
//         self.layers = nn.Sequential(
//             nn.Linear(10, 128),
//             nn.ReLU(),
//             nn.Linear(128, 64),
//             nn.ReLU(),
//             nn.Linear(64, num_classes)
//         )
        
//     def forward(self, x):
//         return self.layers(x)
// `
//     }
//   } else {
//     // Custom model
//     modelDefinition += `class ${getModelClassName(config)}(nn.Module):
//     def __init__(self, num_classes=10):
//         super(${getModelClassName(config)}, self).__init__()
//         # Custom model architecture
// `

//     if (config.customLayers && config.customLayers.length > 0) {
//       // Group layers by type for cleaner code
//       const layerGroups: Record<string, any[]> = {}
//       config.customLayers.forEach((layer) => {
//         if (!layerGroups[layer.category]) {
//           layerGroups[layer.category] = []
//         }
//         layerGroups[layer.category].push(layer)
//       })

//       // Add layers by group
//       Object.entries(layerGroups).forEach(([category, layers]) => {
//         modelDefinition += `        # ${category}\n`
//         layers.forEach((layer, idx) => {
//           const layerName = layer.type.split("(")[0].split(".").pop()
//           modelDefinition += `        self.${layerName.toLowerCase()}${idx + 1} = ${layer.type}\n`
//         })
//         modelDefinition += "\n"
//       })

//       // Create forward method
//       modelDefinition += `    def forward(self, x):\n`
//       config.customLayers.forEach((layer, idx) => {
//         const layerName = layer.type.split("(")[0].split(".").pop()
//         modelDefinition += `        x = self.${layerName?.toLowerCase()}${idx + 1}(x)\n`
//       })
//       modelDefinition += `        return x\n`
//     } else {
//       // Default implementation if no custom layers defined
//       modelDefinition += `        # Default implementation since no custom layers were defined
//         self.layers = nn.Sequential(
//             nn.Linear(10, 128),
//             nn.ReLU(),
//             nn.Linear(128, 64),
//             nn.ReLU(),
//             nn.Linear(64, num_classes)
//         )
        
//     def forward(self, x):
//         return self.layers(x)
// `
//     }
//   }

//   // Training configuration
//   let trainingConfig = `
// # Training Configuration
// def train_model(model, train_loader, val_loader, num_epochs=${config.training?.epochs || 10}, learning_rate=${config.training?.learningRate || 0.001}, weight_decay=${config.training?.weightDecay || 0.0001}):
//     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
//     model = model.to(device)
    
//     # Define loss function
// `

//   // Add loss function
//   if (config.loss && config.loss.name) {
//     trainingConfig += `    criterion = ${config.loss.name}\n`
//   } else {
//     trainingConfig += `    criterion = nn.CrossEntropyLoss()\n`
//   }

//   // Add optimizer
//   trainingConfig += `    
//     # Define optimizer
// `
//   if (config.optimizer && config.optimizer.name) {
//     trainingConfig += `    optimizer = ${config.optimizer.name}\n`
//   } else {
//     trainingConfig += `    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)\n`
//   }

//   // Add scheduler
//   trainingConfig += `    
//     # Define learning rate scheduler
// `
//   if (config.training?.scheduler && config.training.scheduler !== "None") {
//     trainingConfig += `    scheduler = ${config.training.scheduler}(optimizer, ${Object.entries(
//       config.training.schedulerParams || {},
//     )
//       .map(([k, v]) => `${k}=${v}`)
//       .join(", ")})\n`
//   } else {
//     trainingConfig += `    # No scheduler selected\n    scheduler = None\n`
//   }

//   // Add early stopping
//   if (config.training?.earlyStoppingEnabled) {
//     trainingConfig += `
//     # Early stopping parameters
//     patience = ${config.training.earlyStoppingPatience || 5}
//     min_delta = ${config.training.earlyStoppingMinDelta || 0.0001}
//     counter = 0
//     best_loss = float('inf')
// `
//   }

//   // Add metrics
//   trainingConfig += `
//     # Define metrics
// `
//   if (config.metrics && config.metrics.name) {
//     trainingConfig += `    metric = ${config.metrics.name}\n`
//   } else {
//     trainingConfig += `    # No specific metric selected, using loss for monitoring\n`
//   }

//   // Training loop
//   trainingConfig += `
//     # Training loop
//     train_losses = []
//     val_losses = []
    
//     for epoch in range(num_epochs):
//         model.train()
//         running_loss = 0.0
        
//         # Progress bar for training
//         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
//         for inputs, labels in train_pbar:
//             inputs, labels = inputs.to(device), labels.to(device)
            
//             # Zero the parameter gradients
//             optimizer.zero_grad()
            
//             # Forward pass
//             outputs = model(inputs)
//             loss = criterion(

//   // Finish the training loop
//   trainingConfig += \`outputs, labels)
            
//             # Backward pass and optimize
//             loss.backward()
//             optimizer.step()
            
//             # Update statistics
//             running_loss += loss.item() * inputs.size(0)
//             train_pbar.set_postfix({"loss": loss.item()})
        
//         epoch_train_loss = running_loss / len(train_loader.dataset)
//         train_losses.append(epoch_train_loss)
        
//         # Validation
//         model.eval()
//         running_loss = 0.0
        
//         # Progress bar for validation
//         val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
//         with torch.no_grad():
//             for inputs, labels in val_pbar:
//                 inputs, labels = inputs.to(device), labels.to(device)
                
//                 # Forward pass
//                 outputs = model(inputs)
//                 loss = criterion(outputs, labels)
                
//                 # Update statistics
//                 running_loss += loss.item() * inputs.size(0)
//                 val_pbar.set_postfix({"loss": loss.item()})
        
//         epoch_val_loss = running_loss / len(val_loader.dataset)
//         val_losses.append(epoch_val_loss)
        
//         print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
//         # Update learning rate scheduler if defined
//         if scheduler is not None:
//             if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
//                 scheduler.step(epoch_val_loss)
//             else:
//                 scheduler.step()
// `

//   // Add early stopping logic if enabled
//   if (config.training?.earlyStoppingEnabled) {
//     trainingConfig += `
//         # Early stopping
//         if epoch_val_loss < best_loss - min_delta:
//             best_loss = epoch_val_loss
//             counter = 0
//             # Save the best model
//             torch.save(model.state_dict(), 'best_model.pth')
//         else:
//             counter += 1
//             if counter >= patience:
//                 print(f"Early stopping triggered after {epoch+1} epochs")
//                 break
// `
//   }

//   trainingConfig += `
//     # Plot training and validation loss
//     plt.figure(figsize=(10, 5))
//     plt.plot(train_losses, label='Training Loss')
//     plt.plot(val_losses, label='Validation Loss')
//     plt.xlabel('Epochs')
//     plt.ylabel('Loss')
//     plt.title('Training and Validation Loss')
//     plt.legend()
//     plt.savefig('loss_curve.png')
//     plt.close()
    
//     return model, train_losses, val_losses
// `

//   // Evaluation
//   const evaluation = `
// # Evaluation
// def evaluate_model(model, test_loader):
//     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
//     model = model.to(device)
//     model.eval()
    
//     # Placeholder for predictions and targets
//     all_preds = []
//     all_targets = []
    
//     with torch.no_grad():
//         for inputs, labels in test_loader:
//             inputs, labels = inputs.to(device), labels.to(device)
            
//             # Forward pass
//             outputs = model(inputs)
            
//             # Get predictions
//             _, preds = torch.max(outputs, 1)
            
//             # Store predictions and targets
//             all_preds.extend(preds.cpu().numpy())
//             all_targets.extend(labels.cpu().numpy())
    
//     # Calculate metrics
//     accuracy = sum(1 for x, y in zip(all_preds, all_targets) if x == y) / len(all_targets)
//     print(f"Test Accuracy: {accuracy:.4f}")
    
//     # TODO: Add more metrics based on the task
    
//     return accuracy
// `

//   // Inference example
//   const inference = `
// # Inference Example
// def inference(model, input_data):
//     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
//     model = model.to(device)
//     model.eval()
    
//     # Preprocess input data
//     transform = get_transforms()
//     if transform:
//         input_data = transform(input_data)
    
//     # Add batch dimension if needed
//     if input_data.dim() == 3:
//         input_data = input_data.unsqueeze(0)
    
//     input_data = input_data.to(device)
    
//     # Perform inference
//     with torch.no_grad():
//         output = model(input_data)
    
//     # Process output based on task
//     if "${config.subTask}" == "Image Classification" or "${config.subTask}" == "Text Classification":
//         # Get class prediction
//         _, pred = torch.max(output, 1)
//         return pred.item()
//     else:
//         # Return raw output for other tasks
//         return output.cpu().numpy()
// `

//   // Main execution
//   const mainExecution = `
// if __name__ == "__main__":
//     # Set random seed for reproducibility
//     torch.manual_seed(42)
    
//     # Define paths
//     data_path = "path/to/your/data"  # TODO: Replace with actual data path
    
//     # Create data loaders
//     train_loader, val_loader = create_data_loaders(data_path, batch_size=${config.training?.batchSize || 32})
    
//     # Create model
//     model = ${getModelClassName(config)}(num_classes=10)  # TODO: Set appropriate number of classes
    
//     # Train model
//     model, train_losses, val_losses = train_model(
//         model, 
//         train_loader, 
//         val_loader, 
//         num_epochs=${config.training?.epochs || 10}, 
//         learning_rate=${config.training?.learningRate || 0.001}, 
//         weight_decay=${config.training?.weightDecay || 0.0001}
//     )
    
//     # Evaluate model
//     accuracy = evaluate_model(model, val_loader)
    
//     # Save model
//     torch.save(model.state_dict(), "model.pth")
//     print("Model saved to model.pth")
    
//     # Example of loading the model
//     loaded_model = ${getModelClassName(config)}(num_classes=10)
//     loaded_model.load_state_dict(torch.load("model.pth"))
    
//     # Example inference
//     # TODO: Replace with actual input data
//     dummy_input = torch.randn(3, 224, 224)  # Example for image data
//     prediction = inference(loaded_model, dummy_input)
//     print(f"Prediction: {prediction}")
// `

//   // Combine all sections
//   return imports + dataLoading + modelDefinition + trainingConfig + evaluation + inference + mainExecution
// }

// function getModelClassName(config: ModelConfig): string {
//   if (config.pretrainedModel) {
//     return `Pretrained${config.pretrainedModel.replace(/[^a-zA-Z0-9]/g, "")}Model`
//   } else if (config.subTask) {
//     return `${config.subTask.replace(/[^a-zA-Z0-9]/g, "")}Model`
//   } else {
//     return "CustomModel"
//   }
// }

// function getPretrainedModelCode(modelName: string): string {
//   // Map model names to their torchvision equivalents
//   const modelMap: Record<string, string> = {
//     ResNet: "resnet50(pretrained=True)",
//     VGG: "vgg16(pretrained=True)",
//     Inception: "inception_v3(pretrained=True)",
//     EfficientNet: "efficientnet_b0(pretrained=True)",
//     DenseNet: "densenet121(pretrained=True)",
//     MobileNet: "mobilenet_v2(pretrained=True)",
//     AlexNet: "alexnet(pretrained=True)",
//   }

//   return modelMap[modelName] || "resnet50(pretrained=True)"
// }
