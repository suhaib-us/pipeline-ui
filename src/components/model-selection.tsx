"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { PlusCircle, Trash2 } from "lucide-react"
import type { ModelConfig } from "./model-builder"
import { Card, CardContent } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { motion, AnimatePresence } from "framer-motion"

interface ModelSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

// Pretrained model data
const pretrainedModels = {
  "Image Classification": ["ResNet", "VGG", "Inception", "EfficientNet", "DenseNet", "MobileNet", "AlexNet"],
  "Object Detection": [
    "Faster R-CNN",
    "YOLOv5",
    "SSD (Single Shot Multibox Detector)",
    "RetinaNet",
    "Mask R-CNN",
  ],
  "Image Segmentation": ["U-Net", "DeepLabV3", "FCN (Fully Convolutional Network)", "Mask R-CNN"],
  "Natural Language Processing": [
    "BERT",
    "RoBERTa",
    "GPT-2",
    "DistilBERT",
    "T5 (Text-To-Text Transfer Transformer)",
    "XLNet",
  ],
  "Text Classification": ["BERT", "DistilBERT", "ALBERT", "XLNet", "RoBERTa"],
  "Object Tracking": [
    "DeepSORT",
    "SORT (Simple Online and Realtime Tracking)",
    "Track R-CNN",
  ],
  "Audio Processing": ["Wav2Vec2.0", "HuBERT", "Tacotron", "DeepSpeech"],
  "Generative Models": [
    "GAN (Generative Adversarial Network)",
    "DCGAN (Deep Convolutional GAN)",
    "StyleGAN",
    "VQ-VAE (Vector Quantized Variational Autoencoder)",
    "BigGAN",
  ],
  "Reinforcement Learning": [
    "DQN (Deep Q Network)",
    "PPO (Proximal Policy Optimization)",
    "A3C (Asynchronous Advantage Actor-Critic)",
    "SAC (Soft Actor-Critic)",
    "DDPG (Deep Deterministic Policy Gradient)",
  ],
  "Transfer Learning": [
    "ResNet (with transfer learning)",
    "VGG (with transfer learning)",
    "MobileNet (with transfer learning)",
  ],
  "Miscellaneous Tasks": [
    "Time Series Forecasting (Temporal Convolutional Networks, LSTM)",
    "Anomaly Detection (Autoencoders, Isolation Forest)",
    "Recommendation Systems (Neural Collaborative Filtering, Matrix Factorization)",
  ],
}

// Custom model layer categories
const modelLayers = {
  "Input Layers": ["Input", "Embedding"],
  "Convolutional Layers": ["Conv1D", "Conv2D", "Conv3D", "DepthwiseConv2D", "SeparableConv2D"],
  "Pooling Layers": [
    "MaxPooling1D",
    "MaxPooling2D",
    "AveragePooling1D",
    "AveragePooling2D",
    "GlobalMaxPooling",
    "GlobalAveragePooling",
  ],
  "Recurrent Layers": ["LSTM", "GRU", "SimpleRNN", "Bidirectional RNN"],
  "Dense Layers": ["Dense", "Dropout", "BatchNormalization", "ActivityRegularization"],
  "Attention Layers": ["Attention", "MultiHeadAttention", "SelfAttention"],
  "Activation Layers": ["ReLU", "Sigmoid", "Softmax", "Tanh"],
  "Normalization Layers": ["BatchNormalization", "LayerNormalization", "InstanceNormalization"],
  "Miscellaneous Layers": ["Add", "Multiply", "Concatenate", "Reshape", "Flatten", "Permute", "Lambda (Custom Layer)"],
}

export function ModelSelection({ config, updateConfig }: ModelSelectionProps) {
  const [selectedLayerCategory, setSelectedLayerCategory] = useState("")
  const [selectedLayer, setSelectedLayer] = useState("")
  const [availableLayers, setAvailableLayers] = useState<string[]>([])
  const [modelCategory, setModelCategory] = useState(config.pretrainedModelCategory || "")
  const [availableModels, setAvailableModels] = useState<string[]>([])

  const handleTabChange = (value: string) => {
    updateConfig({ modelType: value as "pretrained" | "custom" })
  }

  const handleLayerCategoryChange = (category: string) => {
    setSelectedLayerCategory(category)
    setAvailableLayers(modelLayers[category as keyof typeof modelLayers] || [])
    setSelectedLayer("")
  }

  const handleModelCategoryChange = (category: string) => {
    setModelCategory(category)
    setAvailableModels(pretrainedModels[category as keyof typeof pretrainedModels] || [])
    updateConfig({
      pretrainedModelCategory: category,
      pretrainedModel: "",
    })
  }

  const handleModelChange = (model: string) => {
    updateConfig({ pretrainedModel: model })
  }

  const addLayer = () => {
    if (selectedLayerCategory && selectedLayer) {
      const newLayer = {
        type: selectedLayer,
        category: selectedLayerCategory,
        config: {},
      }

      updateConfig({
        customLayers: [...(config.customLayers || []), newLayer],
      })

      // Reset selections
      setSelectedLayer("")
    }
  }

  const removeLayer = (index: number) => {
    const updatedLayers = [...(config.customLayers || [])]
    updatedLayers.splice(index, 1)
    updateConfig({ customLayers: updatedLayers })
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <h2 className="text-2xl font-bold text-slate-900">Model Selection</h2>
        <p className="text-slate-500">Choose between pre-trained models or build a custom model</p>
      </motion.div>

      <Tabs defaultValue={config.modelType} onValueChange={handleTabChange} className="w-full">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="pretrained" className="transition-all duration-200 data-[state=active]:shadow-md">
              Pre-trained Models
            </TabsTrigger>
            <TabsTrigger value="custom" className="transition-all duration-200 data-[state=active]:shadow-md">
              Custom Model
            </TabsTrigger>
          </TabsList>
        </motion.div>

        <TabsContent value="pretrained" className="mt-4 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <motion.div
              className="space-y-2"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.2 }}
            >
              <label className="text-sm font-medium text-slate-700">Model Type</label>
              <Select value={modelCategory} onValueChange={handleModelCategoryChange}>
                <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
                  <SelectValue placeholder="Select model type" />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(pretrainedModels).map((category) => (
                    <SelectItem key={category} value={category}>
                      {category}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </motion.div>

            <motion.div
              className="space-y-2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.3 }}
            >
              <label className="text-sm font-medium text-slate-700">Specific Model</label>
              <Select value={config.pretrainedModel} onValueChange={handleModelChange} disabled={!modelCategory}>
                <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {pretrainedModels[modelCategory as keyof typeof pretrainedModels]?.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </motion.div>
          </div>
        </TabsContent>

        <TabsContent value="custom" className="mt-4 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <motion.div
              className="space-y-2"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.2 }}
            >
              <label className="text-sm font-medium text-slate-700">Layer Category</label>
              <Select value={selectedLayerCategory} onValueChange={handleLayerCategoryChange}>
                <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
                  <SelectValue placeholder="Select layer category" />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(modelLayers).map((category) => (
                    <SelectItem key={category} value={category}>
                      {category}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </motion.div>

            <motion.div
              className="space-y-2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.3 }}
            >
              <label className="text-sm font-medium text-slate-700">Layer Type</label>
              <Select value={selectedLayer} onValueChange={setSelectedLayer} disabled={!selectedLayerCategory}>
                <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
                  <SelectValue placeholder="Select a layer" />
                </SelectTrigger>
                <SelectContent>
                  {availableLayers.map((layer) => (
                    <SelectItem key={layer} value={layer}>
                      {layer}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </motion.div>
          </div>

          <motion.div
            className="flex justify-end"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.4 }}
          >
            <Button
              onClick={addLayer}
              disabled={!selectedLayer}
              className="flex items-center gap-1 transition-all duration-200 hover:scale-105"
            >
              <PlusCircle className="h-4 w-4" />
              Add Layer
            </Button>
          </motion.div>

          <motion.div
            className="space-y-2"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.5 }}
          >
            <label className="text-sm font-medium text-slate-700">Model Architecture</label>
            <Card className="overflow-hidden border-slate-200 shadow-sm transition-all duration-200 hover:shadow-md">
              <CardContent className="p-4">
                {config.customLayers && config.customLayers.length > 0 ? (
                  <ScrollArea className="h-[200px]">
                    <AnimatePresence>
                      <div className="space-y-2">
                        {config.customLayers.map((layer, index) => (
                          <motion.div
                            key={index}
                            className="flex items-center justify-between rounded-md border border-slate-200 bg-slate-50 p-3"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                            transition={{ duration: 0.2 }}
                            layout
                          >
                            <div>
                              <span className="font-medium">{layer.type}</span>
                              <span className="ml-2 text-xs text-slate-500">({layer.category})</span>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => removeLayer(index)}
                              className="h-8 w-8 p-0 text-slate-500 transition-colors duration-200 hover:bg-red-50 hover:text-red-500"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </motion.div>
                        ))}
                      </div>
                    </AnimatePresence>
                  </ScrollArea>
                ) : (
                  <div className="flex h-[200px] items-center justify-center text-slate-500">
                    No layers added yet. Start building your model by adding layers.
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
