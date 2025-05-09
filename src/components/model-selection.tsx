import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { PlusCircle, Trash2 } from "lucide-react"
import type { ModelConfig } from "./model-builder"
import { Card, CardContent } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { motion, AnimatePresence } from "framer-motion"
import { Input } from "@/components/ui/input"

interface ModelSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

// Pretrained model data based on task
const pretrainedModels: Record<string, Record<string, string[]>> = {
  dl: {
    "Image Classification": [
      "ResNet (18, 34, 50, 101, 152)",
      "VGG (11, 13, 16, 19)",
      "Inception v3",
      "EfficientNet (B0-B7)",
      "DenseNet (121, 169, 201, 161)",
      "MobileNet v2/v3",
      "ViT (Vision Transformer)",
      "ConvNeXt",
    ],
    "Object Detection": [
      "Faster R-CNN",
      "SSD (Single Shot MultiBox Detector)",
      "RetinaNet",
      "YOLOv5 (via PyTorch Hub)",
      "DETR (Detection Transformer)",
    ],
    "Image Segmentation": ["U-Net", "DeepLabV3", "FCN (Fully Convolutional Network)", "Mask R-CNN", "SegFormer"],
    "Text Classification": ["BERT", "RoBERTa", "DistilBERT", "XLNet", "ALBERT"],
    "Sentiment Analysis": ["BERT", "RoBERTa", "DistilBERT", "XLNet", "ALBERT"],
    "Named Entity Recognition": ["BERT", "RoBERTa", "DistilBERT", "XLNet", "ALBERT"],
    "Text Generation": ["GPT-2", "T5", "BART", "XLNet", "CTRL"],
    "Machine Translation": ["T5", "BART", "MarianMT", "M2M100"],
    "Text Summarization": ["T5", "BART", "Pegasus", "ProphetNet"],
    "Speech Recognition": ["Wav2Vec2.0", "HuBERT", "Whisper", "DeepSpeech"],
    "Audio Classification": ["Wav2Vec2.0", "HuBERT", "PANNs (Pretrained Audio Neural Networks)"],
    "Audio Generation": ["Tacotron", "WaveNet", "HiFi-GAN"],
    "Voice Conversion": ["AutoVC", "SpeechSplit", "StarGAN-VC"],
    "GAN (Generative Adversarial Networks)": ["DCGAN", "StyleGAN", "CycleGAN", "Pix2Pix", "BigGAN"],
    "VAE (Variational Autoencoders)": ["VQ-VAE", "Beta-VAE", "NVAE"],
    "Diffusion Models": [
      "DDPM (Denoising Diffusion Probabilistic Models)",
      "DDIM (Denoising Diffusion Implicit Models)",
      "Stable Diffusion",
    ],
    "Autoregressive Models": ["PixelCNN", "PixelRNN", "Transformer-XL"],
    "Value-based Methods (DQN, etc.)": ["DQN (Deep Q Network)", "Double DQN", "Dueling DQN", "Rainbow DQN"],
    "Policy-based Methods (PPO, etc.)": [
      "PPO (Proximal Policy Optimization)",
      "TRPO (Trust Region Policy Optimization)",
      "A2C (Advantage Actor-Critic)",
      "SAC (Soft Actor-Critic)",
    ],
    "Model-based Methods": ["World Models", "MuZero", "Dreamer"],
    "Multi-agent Systems": ["MADDPG (Multi-Agent DDPG)", "QMIX", "MAPPO (Multi-Agent PPO)"],
    "Anomaly Detection": ["Autoencoder", "Isolation Forest", "One-Class SVM"],
    "Dimensionality Reduction": ["PCA (Principal Component Analysis)", "t-SNE", "UMAP", "Autoencoder"],
    "Recommendation Systems": ["Neural Collaborative Filtering", "Matrix Factorization", "Wide & Deep"],
    "Time Series Analysis": ["LSTM", "GRU", "Transformer", "TCN (Temporal Convolutional Network)"],
  },
  ml: {
    "Binary Classification": ["Logistic Regression", "SVM", "Random Forest", "Gradient Boosting", "Neural Network"],
    "Multi-class Classification": [
      "Logistic Regression",
      "SVM",
      "Random Forest",
      "Gradient Boosting",
      "Neural Network",
    ],
    "Multi-label Classification": ["Binary Relevance", "Classifier Chains", "Label Powerset", "Neural Network"],
    "Simple Regression": ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet", "Neural Network"],
    "Multiple Regression": [
      "Linear Regression",
      "Ridge Regression",
      "Lasso Regression",
      "ElasticNet",
      "Neural Network",
    ],
    "Polynomial Regression": [
      "Polynomial Features + Linear Regression",
      "SVR with Polynomial Kernel",
      "Neural Network",
    ],
    "Time Series Forecasting": ["ARIMA", "Prophet", "LSTM", "GRU", "Neural Network"],
    "K-Means": ["K-Means", "K-Means++", "Mini-Batch K-Means"],
    "Hierarchical Clustering": ["Agglomerative Clustering", "Divisive Clustering"],
    DBSCAN: ["DBSCAN", "HDBSCAN"],
    "Gaussian Mixture Models": ["GMM", "Bayesian GMM"],
  },
}

// Custom model layer categories
const modelLayers = {
  "Input Layers": ["torch.nn.Linear(in_features, out_features)", "torch.nn.Embedding(num_embeddings, embedding_dim)"],
  "Convolutional Layers": [
    "torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)",
    "torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)",
    "torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)",
    "torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)",
    "torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)",
    "torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)",
  ],
  "Pooling Layers": [
    "torch.nn.MaxPool1d(kernel_size, stride, padding)",
    "torch.nn.MaxPool2d(kernel_size, stride, padding)",
    "torch.nn.MaxPool3d(kernel_size, stride, padding)",
    "torch.nn.AvgPool1d(kernel_size, stride, padding)",
    "torch.nn.AvgPool2d(kernel_size, stride, padding)",
    "torch.nn.AvgPool3d(kernel_size, stride, padding)",
    "torch.nn.AdaptiveMaxPool1d(output_size)",
    "torch.nn.AdaptiveMaxPool2d(output_size)",
    "torch.nn.AdaptiveAvgPool1d(output_size)",
    "torch.nn.AdaptiveAvgPool2d(output_size)",
  ],
  "Recurrent Layers": [
    "torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional)",
    "torch.nn.GRU(input_size, hidden_size, num_layers, bidirectional)",
    "torch.nn.RNN(input_size, hidden_size, num_layers, bidirectional)",
  ],
  "Transformer Layers": [
    "torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)",
    "torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)",
    "torch.nn.MultiheadAttention(embed_dim, num_heads)",
  ],
  "Normalization Layers": [
    "torch.nn.BatchNorm1d(num_features)",
    "torch.nn.BatchNorm2d(num_features)",
    "torch.nn.BatchNorm3d(num_features)",
    "torch.nn.LayerNorm(normalized_shape)",
    "torch.nn.InstanceNorm1d(num_features)",
    "torch.nn.InstanceNorm2d(num_features)",
    "torch.nn.GroupNorm(num_groups, num_channels)",
  ],
  "Activation Layers": [
    "torch.nn.ReLU()",
    "torch.nn.LeakyReLU(negative_slope)",
    "torch.nn.GELU()",
    "torch.nn.Sigmoid()",
    "torch.nn.Tanh()",
    "torch.nn.Softmax(dim)",
    "torch.nn.SiLU()",
    "torch.nn.Mish()",
  ],
  "Regularization Layers": ["torch.nn.Dropout(p)", "torch.nn.Dropout2d(p)", "torch.nn.Dropout3d(p)"],
  "Reshape Layers": ["torch.nn.Flatten(start_dim, end_dim)", "torch.nn.Unflatten(dim, unflattened_size)"],
  "Custom Layer": ["Lambda layer for custom operations"],
}

export function ModelSelection({ config, updateConfig }: ModelSelectionProps) {
  const [selectedLayerCategory, setSelectedLayerCategory] = useState("")
  const [selectedLayer, setSelectedLayer] = useState("")
  const [availableLayers, setAvailableLayers] = useState<string[]>([])
  const [modelCategory, setModelCategory] = useState(config.pretrainedModelCategory || "")
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [layerParams, setLayerParams] = useState<Record<string, string>>({})

  const handleTabChange = (value: string) => {
    updateConfig({ modelType: value as "pretrained" | "custom" })
  }

  const handleLayerCategoryChange = (category: string) => {
    setSelectedLayerCategory(category)
    setAvailableLayers(modelLayers[category as keyof typeof modelLayers] || [])
    setSelectedLayer("")
    setLayerParams({})
  }

  const handleLayerChange = (layer: string) => {
    setSelectedLayer(layer)

    // Extract parameters from layer string
    const paramMatch = layer.match(/$$(.*?)$$/)
    if (paramMatch && paramMatch[1]) {
      const params = paramMatch[1].split(",").map((p) => p.trim())
      const paramObj: Record<string, string> = {}

      params.forEach((param) => {
        const [name] = param.split("=")
        paramObj[name] = ""
      })

      setLayerParams(paramObj)
    } else {
      setLayerParams({})
    }
  }

  const handleParamChange = (param: string, value: string) => {
    setLayerParams((prev) => ({
      ...prev,
      [param]: value,
    }))
  }

  const handleModelCategoryChange = (category: string) => {
    setModelCategory(category)

    if (config.mode && config.subTask) {
      const models = pretrainedModels[config.mode]?.[config.subTask] || []
      setAvailableModels(models)
    }

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
        config: { ...layerParams },
      }

      updateConfig({
        customLayers: [...(config.customLayers || []), newLayer],
      })

      // Reset selections
      setSelectedLayer("")
      setLayerParams({})
    }
  }

  const removeLayer = (index: number) => {
    const updatedLayers = [...(config.customLayers || [])]
    updatedLayers.splice(index, 1)
    updateConfig({ customLayers: updatedLayers })
  }

  useEffect(() => {
    if (config.mode && config.subTask) {
      const models = pretrainedModels[config.mode]?.[config.subTask] || []
      setAvailableModels(models)

      // Reset model if task changes
      if (config.pretrainedModel && !models.includes(config.pretrainedModel)) {
        updateConfig({ pretrainedModel: "" })
      }
    } else {
      setAvailableModels([])
    }
  }, [config.mode, config.subTask])

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
          {config.mode && config.subTask ? (
            <div className="grid gap-6 md:grid-cols-1">
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <label className="text-sm font-medium text-slate-700">Select a Model</label>
                <Select value={config.pretrainedModel} onValueChange={handleModelChange}>
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((model) => (
                      <SelectItem key={model} value={model}>
                        {model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {config.pretrainedModel && (
                  <motion.p
                    className="mt-2 text-sm text-slate-500"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    {getModelDescription(config.pretrainedModel)}
                  </motion.p>
                )}
              </motion.div>
            </div>
          ) : (
            <motion.div
              className="rounded-lg border border-dashed border-slate-300 p-8 text-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <p className="text-slate-500">Please select a framework and task first to see available models.</p>
            </motion.div>
          )}
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
              <Select value={selectedLayer} onValueChange={handleLayerChange} disabled={!selectedLayerCategory}>
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

          {selectedLayer && Object.keys(layerParams).length > 0 && (
            <motion.div
              className="space-y-4 rounded-lg border border-slate-200 bg-slate-50 p-4"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <h3 className="text-sm font-medium text-slate-700">Layer Parameters</h3>
              <div className="grid gap-4 md:grid-cols-2">
                {Object.keys(layerParams).map((param) => (
                  <div key={param} className="space-y-1">
                    <label className="text-xs font-medium text-slate-600">{param}</label>
                    <Input
                      value={layerParams[param]}
                      onChange={(e) => handleParamChange(param, e.target.value)}
                      placeholder={`Enter ${param}`}
                      className="h-8 text-sm"
                    />
                  </div>
                ))}
              </div>
            </motion.div>
          )}

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
                  <ScrollArea className="h-[300px]">
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
                              {Object.keys(layer.config).length > 0 && (
                                <div className="mt-1 text-xs text-slate-500">
                                  Params:{" "}
                                  {Object.entries(layer.config)
                                    .map(([key, value]) => `${key}=${value || "default"}`)
                                    .join(", ")}
                                </div>
                              )}
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
                  <div className="flex h-[300px] items-center justify-center text-slate-500">
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

function getModelDescription(model: string): string {
  const descriptions: Record<string, string> = {
    "ResNet (18, 34, 50, 101, 152)":
      "A deep residual learning framework that addresses the degradation problem in deep networks using skip connections.",
    "VGG (11, 13, 16, 19)":
      "A deep convolutional network known for its simplicity using only 3×3 convolutional layers stacked on top of each other.",
    "Inception v3":
      "A convolutional neural network architecture that uses inception modules with factorized convolutions to reduce parameters.",
    "EfficientNet (B0-B7)":
      "A family of models that uniformly scales all dimensions of depth, width, and resolution using a compound coefficient.",
    "DenseNet (121, 169, 201, 161)":
      "A network architecture where each layer is connected to every other layer in a feed-forward fashion to improve information flow.",
    "MobileNet v2/v3":
      "Lightweight models designed for mobile and edge devices using depthwise separable convolutions.",
    "ViT (Vision Transformer)": "Applies the transformer architecture to image patches for image classification tasks.",
    ConvNeXt:
      "A pure convolutional network designed to match the performance of transformers with the efficiency of CNNs.",
    "Faster R-CNN":
      "A region-based convolutional network for object detection that introduces a Region Proposal Network (RPN).",
    "SSD (Single Shot MultiBox Detector)":
      "A method for detecting objects in images using a single deep neural network that predicts bounding boxes and class probabilities.",
    RetinaNet: "An object detection model that addresses class imbalance using Focal Loss.",
    "YOLOv5 (via PyTorch Hub)":
      "You Only Look Once - a real-time object detection system that processes images in a single pass.",
    "DETR (Detection Transformer)":
      "Detection Transformer that uses a transformer encoder-decoder architecture for object detection.",
    "U-Net":
      "A convolutional network architecture for fast and precise segmentation of images, originally developed for biomedical image segmentation.",
    DeepLabV3:
      "A semantic segmentation architecture that employs atrous convolutions to handle objects at multiple scales.",
    "FCN (Fully Convolutional Network)":
      "A network that extends CNNs to arbitrary-sized inputs for semantic segmentation.",
    "Mask R-CNN": "An extension of Faster R-CNN that adds a branch for predicting segmentation masks.",
    SegFormer: "A transformer-based architecture for semantic segmentation that combines local and global features.",
    BERT: "Bidirectional Encoder Representations from Transformers - a transformer-based model designed for NLP tasks.",
    RoBERTa: "A robustly optimized BERT pretraining approach with improved training methodology.",
    DistilBERT: "A distilled version of BERT that is smaller, faster, and retains most of BERT's performance.",
    "GPT-2": "Generative Pretrained Transformer 2 - an autoregressive language model for text generation.",
    T5: "Text-to-Text Transfer Transformer - a unified framework that converts all NLP tasks to a text-to-text format.",
    XLNet:
      "A generalized autoregressive pretraining method that overcomes limitations of BERT by using permutation language modeling.",
    BART: "A denoising autoencoder for pretraining sequence-to-sequence models.",
    "Wav2Vec2.0": "A framework for self-supervised learning of speech representations.",
    HuBERT: "Hidden-Unit BERT - a self-supervised speech representation learning approach.",
    Whisper: "A robust speech recognition model trained on a large dataset of diverse audio.",
    DeepSpeech: "An end-to-end speech recognition model using recurrent neural networks.",
    Tacotron: "An end-to-end text-to-speech synthesis model.",
    DCGAN: "Deep Convolutional GAN - a class of CNNs for generating realistic images.",
    StyleGAN: "A style-based generator architecture for GANs that offers control over the image synthesis process.",
    CycleGAN: "A model for unpaired image-to-image translation using cycle-consistent adversarial networks.",
    Pix2Pix: "A conditional GAN for image-to-image translation tasks with paired data.",
    BigGAN: "A large-scale GAN architecture for high-fidelity image synthesis.",
    "VQ-VAE": "Vector Quantized Variational Autoencoder - a model that combines VAEs with vector quantization.",
    "Beta-VAE": "A modification of the VAE framework that learns disentangled latent representations.",
    NVAE: "Nouveau VAE - a deep hierarchical VAE with a carefully designed architecture.",
    "DDPM (Denoising Diffusion Probabilistic Models)":
      "A class of generative models that generate samples by gradually denoising a signal.",
    "DDIM (Denoising Diffusion Implicit Models)":
      'An improved sampling method for diffusion models that accelerates the sampling process.",celerates the sampling process.',
    "Stable Diffusion": "A latent diffusion model for high-quality image generation with controllable conditions.",
    PixelCNN: "An autoregressive model that generates images pixel by pixel using masked convolutions.",
    PixelRNN: "An autoregressive model that uses RNNs to generate images pixel by pixel.",
    "Transformer-XL":
      "An extension of the Transformer model with a segment-level recurrence mechanism for longer context.",
    "DQN (Deep Q Network)": "A reinforcement learning algorithm that combines Q-learning with deep neural networks.",
    "Double DQN": "An improvement to DQN that addresses overestimation bias in Q-learning.",
    "Dueling DQN": "A DQN architecture that separates state value and advantage functions.",
    "Rainbow DQN": "A combination of multiple improvements to DQN for better performance.",
    "PPO (Proximal Policy Optimization)": "A policy gradient method that uses a clipped surrogate objective.",
    "TRPO (Trust Region Policy Optimization)": "A policy optimization method that ensures small policy updates.",
    "A2C (Advantage Actor-Critic)": "A synchronous version of the Asynchronous Advantage Actor-Critic algorithm.",
    "SAC (Soft Actor-Critic)": "An off-policy actor-critic algorithm that maximizes both expected return and entropy.",
    "World Models":
      "A model-based reinforcement learning approach that learns a compressed spatial and temporal representation of the environment.",
    MuZero:
      "A model-based reinforcement learning algorithm that plans with a learned model without modeling the environment dynamics directly.",
    Dreamer: "A world model that learns to imagine and plan in a latent space for efficient reinforcement learning.",
    "MADDPG (Multi-Agent DDPG)":
      "An extension of DDPG for multi-agent environments with centralized training and decentralized execution.",
    QMIX: "A value-based method for multi-agent reinforcement learning that enforces monotonicity constraints.",
    "MAPPO (Multi-Agent PPO)": "A multi-agent version of PPO with centralized training and decentralized execution.",
    Autoencoder: "A neural network that learns to encode data into a latent representation and decode it back.",
    "Isolation Forest": "An algorithm that isolates observations by randomly selecting a feature and a split value.",
    "One-Class SVM": "A support vector machine variant for novelty detection.",
    "PCA (Principal Component Analysis)":
      "A dimensionality reduction technique that finds the directions of maximum variance.",
    "t-SNE":
      "t-Distributed Stochastic Neighbor Embedding - a technique for dimensionality reduction that preserves local structure.",
    UMAP: "Uniform Manifold Approximation and Projection - a dimensionality reduction technique that preserves both local and global structure.",
    "Neural Collaborative Filtering":
      "A neural network approach to collaborative filtering for recommendation systems.",
    "Matrix Factorization":
      "A technique that factorizes the user-item interaction matrix into lower-dimensional matrices.",
    "Wide & Deep": "A model that combines a wide linear model and a deep neural network for recommendation systems.",
    LSTM: "Long Short-Term Memory - a type of RNN designed to handle long-term dependencies.",
    GRU: "Gated Recurrent Unit - a simplified version of LSTM with fewer parameters.",
    Transformer: "A model architecture that relies entirely on attention mechanisms for sequence modeling.",
    "TCN (Temporal Convolutional Network)":
      "A convolutional architecture for sequence modeling that can handle long-range patterns.",
    "Logistic Regression": "A statistical model that uses a logistic function to model a binary dependent variable.",
    SVM: "Support Vector Machine - a supervised learning model for classification and regression tasks.",
    "Random Forest": "An ensemble learning method that constructs multiple decision trees during training.",
    "Gradient Boosting":
      "An ensemble technique that builds models sequentially, with each new model correcting errors of the previous ones.",
    "Neural Network": "A computational model inspired by the structure and function of biological neural networks.",
  }

  return descriptions[model] || "A pre-trained model for your specific task."
}
