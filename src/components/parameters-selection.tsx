"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ModelConfig } from "./model-builder"

interface ParametersSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

// Monitoring and metrics data
const monitoringData = {
  "Training Monitoring": ["Loss Tracking", "Metric Tracking (Accuracy, F1, etc.)", "Learning Rate Scheduling"],
  "Performance Profiling": ["Memory Usage", "GPU Utilization", "Execution Time"],
  "Experiment Tracking": ["Parameter Logging (hyperparameters)", "Model Checkpointing", "Versioning"],
  "Logging Tools": ["TensorBoard", "Weights & Biases", "MLflow", "Neptune.ai"],
  Alerting: ["Threshold-Based Alerts (e.g., accuracy drop)", "Resource Alerts (e.g., high memory usage)"],
}

// Optimizers data
const optimizersData = {
  "Gradient Descent-Based": ["SGD (Stochastic Gradient Descent)", "Adam", "RMSprop", "AdaGrad", "AdaDelta", "AdamW"],
  "Second-Order Methods": ["LBFGS", "Newton-CG (External implementation)"],
  "Adaptive Methods": ["Adam", "AdamW", "AMSGrad", "RAdam (Rectified Adam)"],
  "Regularization-Based": ["Adamax", "Nadam (Nesterov-accelerated Adam)"],
  "Other Optimizers": [
    "ASGD (Averaged Stochastic Gradient Descent)",
    "SparseAdam (for sparse tensors)",
    "FTRL (Follow the Regularized Leader, External implementation)",
  ],
}

// Loss functions data
const lossesData = {
  "Regression Losses": ["MSELoss (Mean Squared Error)", "L1Loss (Mean Absolute Error)", "HuberLoss (Smooth L1 Loss)"],
  "Classification Losses": [
    "CrossEntropyLoss",
    "BCEWithLogitsLoss (Binary Cross Entropy with Logits)",
    "NLLLoss (Negative Log Likelihood)",
    "HingeEmbeddingLoss",
  ],
  "Embedding Losses": ["TripletMarginLoss", "CosineEmbeddingLoss"],
  "Segmentation Losses": [
    "DiceLoss (External or custom implementation)",
    "JaccardLoss (External or custom implementation)",
  ],
  "Adversarial Losses": ["GAN Loss (BCE or custom, for GANs)", "Wasserstein Loss (WGAN)"],
  "Customizable Losses": ["Custom Loss (Lambda layer for user-defined functions)", "MultiLabelSoftMarginLoss"],
}

// Metrics data
const metricsData = {
  "Classification Metrics": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
  "Regression Metrics": [
    "Mean Absolute Error (MAE)",
    "Mean Squared Error (MSE)",
    "Root Mean Squared Error (RMSE)",
    "R-squared",
  ],
  "Segmentation Metrics": ["Dice Coefficient", "Jaccard Index (IoU - Intersection over Union)"],
  "Ranking Metrics": ["Mean Reciprocal Rank (MRR)", "Normalized Discounted Cumulative Gain (NDCG)"],
  "Clustering Metrics": ["Silhouette Score", "Davies-Bouldin Index", "Adjusted Rand Index"],
  Miscellaneous: [
    "Log-Loss (Binary Cross-Entropy Loss as Metric)",
    "Cosine Similarity",
    "Mean Squared Logarithmic Error (MSLE)",
  ],
}

export function ParametersSelection({ config, updateConfig }: ParametersSelectionProps) {
  const [monitoringCategory, setMonitoringCategory] = useState("")
  const [optimizerCategory, setOptimizerCategory] = useState("")
  const [lossCategory, setLossCategory] = useState("")
  const [metricCategory, setMetricCategory] = useState("")

  const handleMonitoringCategoryChange = (category: string) => {
    setMonitoringCategory(category)
    updateConfig({
      monitoring: {
        category,
        option: "",
      },
    })
  }

  const handleMonitoringOptionChange = (option: string) => {
    updateConfig({
      monitoring: {
        category: monitoringCategory,
        option,
      },
    })
  }

  const handleOptimizerCategoryChange = (category: string) => {
    setOptimizerCategory(category)
    updateConfig({
      optimizer: {
        category,
        name: "",
      },
    })
  }

  const handleOptimizerChange = (name: string) => {
    updateConfig({
      optimizer: {
        category: optimizerCategory,
        name,
      },
    })
  }

  const handleLossCategoryChange = (category: string) => {
    setLossCategory(category)
    updateConfig({
      loss: {
        category,
        name: "",
      },
    })
  }

  const handleLossChange = (name: string) => {
    updateConfig({
      loss: {
        category: lossCategory,
        name,
      },
    })
  }

  const handleMetricCategoryChange = (category: string) => {
    setMetricCategory(category)
    updateConfig({
      metrics: {
        category,
        name: "",
      },
    })
  }

  const handleMetricChange = (name: string) => {
    updateConfig({
      metrics: {
        category: metricCategory,
        name,
      },
    })
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Parameters</h2>
        <p className="text-slate-500">Configure monitoring, optimizers, loss functions, and evaluation metrics</p>
      </div>

      <Tabs defaultValue="monitoring" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          <TabsTrigger value="optimizers">Optimizers</TabsTrigger>
          <TabsTrigger value="losses">Losses</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        {/* Monitoring Tab */}
        <TabsContent value="monitoring" className="mt-4 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Category</label>
              <Select value={monitoringCategory} onValueChange={handleMonitoringCategoryChange}>
                <SelectTrigger>
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(monitoringData).map((category) => (
                    <SelectItem key={category} value={category}>
                      {category}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Option</label>
              <Select
                value={config.monitoring?.option || ""}
                onValueChange={handleMonitoringOptionChange}
                disabled={!monitoringCategory}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select option" />
                </SelectTrigger>
                <SelectContent>
                  {monitoringData[monitoringCategory as keyof typeof monitoringData]?.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </TabsContent>

        {/* Optimizers Tab */}
        <TabsContent value="optimizers" className="mt-4 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Category</label>
              <Select value={optimizerCategory} onValueChange={handleOptimizerCategoryChange}>
                <SelectTrigger>
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(optimizersData).map((category) => (
                    <SelectItem key={category} value={category}>
                      {category}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Optimizer</label>
              <Select
                value={config.optimizer?.name || ""}
                onValueChange={handleOptimizerChange}
                disabled={!optimizerCategory}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select optimizer" />
                </SelectTrigger>
                <SelectContent>
                  {optimizersData[optimizerCategory as keyof typeof optimizersData]?.map((optimizer) => (
                    <SelectItem key={optimizer} value={optimizer}>
                      {optimizer}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </TabsContent>

        {/* Losses Tab */}
        <TabsContent value="losses" className="mt-4 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Category</label>
              <Select value={lossCategory} onValueChange={handleLossCategoryChange}>
                <SelectTrigger>
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(lossesData).map((category) => (
                    <SelectItem key={category} value={category}>
                      {category}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Loss Function</label>
              <Select value={config.loss?.name || ""} onValueChange={handleLossChange} disabled={!lossCategory}>
                <SelectTrigger>
                  <SelectValue placeholder="Select loss function" />
                </SelectTrigger>
                <SelectContent>
                  {lossesData[lossCategory as keyof typeof lossesData]?.map((loss) => (
                    <SelectItem key={loss} value={loss}>
                      {loss}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </TabsContent>

        {/* Metrics Tab */}
        <TabsContent value="metrics" className="mt-4 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Category</label>
              <Select value={metricCategory} onValueChange={handleMetricCategoryChange}>
                <SelectTrigger>
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(metricsData).map((category) => (
                    <SelectItem key={category} value={category}>
                      {category}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Metric</label>
              <Select value={config.metrics?.name || ""} onValueChange={handleMetricChange} disabled={!metricCategory}>
                <SelectTrigger>
                  <SelectValue placeholder="Select metric" />
                </SelectTrigger>
                <SelectContent>
                  {metricsData[metricCategory as keyof typeof metricsData]?.map((metric) => (
                    <SelectItem key={metric} value={metric}>
                      {metric}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
