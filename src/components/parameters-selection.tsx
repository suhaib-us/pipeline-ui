"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import type { ModelConfig } from "./model-builder"
import { motion } from "framer-motion"

interface ParametersSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

// Monitoring and metrics data
const monitoringData = {
  "Training Monitoring": ["Loss Tracking", "Metric Tracking (Accuracy, F1, etc.)", "Learning Rate Scheduling"],
  "Performance Profiling": ["Memory Usage", "GPU Utilization", "Execution Time"],
  "Experiment Tracking": ["Parameter Logging (hyperparameters)", "Model Checkpointing", "Versioning"],
  "Logging Tools": ["TensorBoard", "Weights & Biases", "MLflow", "PyTorch Lightning"],
  Alerting: ["Threshold-Based Alerts", "Resource Alerts"],
}

// Optimizers data
const optimizersData = {
  "Gradient Descent-Based": ["torch.optim.SGD(params, lr, momentum, weight_decay, nesterov)"],
  "Adaptive Methods": [
    "torch.optim.Adam(params, lr, betas, eps, weight_decay, amsgrad)",
    "torch.optim.AdamW(params, lr, betas, eps, weight_decay)",
    "torch.optim.Adamax(params, lr, betas, eps, weight_decay)",
    "torch.optim.RMSprop(params, lr, alpha, eps, weight_decay, momentum)",
    "torch.optim.Adagrad(params, lr, lr_decay, weight_decay, eps)",
    "torch.optim.Adadelta(params, lr, rho, eps, weight_decay)",
  ],
  "Second-Order Methods": ["torch.optim.LBFGS(params, lr, max_iter, max_eval, tolerance_grad, tolerance_change)"],
  "Regularization-Based": [
    "torch.optim.NAdam(params, lr, betas, eps, weight_decay, momentum_decay)",
    "torch.optim.RAdam(params, lr, betas, eps, weight_decay)",
  ],
  "Other Optimizers": [
    "torch.optim.SparseAdam(params, lr, betas, eps)",
    "torch.optim.ASGD(params, lr, lambd, alpha, t0, weight_decay)",
  ],
}

// Loss functions data
const lossesData = {
  "Regression Losses": [
    "torch.nn.MSELoss(reduction)",
    "torch.nn.L1Loss(reduction)",
    "torch.nn.SmoothL1Loss(reduction, beta)",
    "torch.nn.HuberLoss(reduction, delta)",
  ],
  "Classification Losses": [
    "torch.nn.CrossEntropyLoss(weight, reduction)",
    "torch.nn.BCELoss(weight, reduction)",
    "torch.nn.BCEWithLogitsLoss(weight, reduction, pos_weight)",
    "torch.nn.NLLLoss(weight, reduction)",
    "torch.nn.MarginRankingLoss(margin, reduction)",
  ],
  "Embedding Losses": [
    "torch.nn.TripletMarginLoss(margin, p, eps, swap, reduction)",
    "torch.nn.CosineEmbeddingLoss(margin, reduction)",
    "torch.nn.MultiMarginLoss(p, margin, weight, reduction)",
  ],
  "Segmentation Losses": [
    "torch.nn.functional.binary_cross_entropy_with_logits (with dice loss implementation)",
    "torch.nn.functional.cross_entropy (with focal loss implementation)",
  ],
  "Adversarial Losses": ["Minimax GAN Loss", "Wasserstein Loss", "Least Squares Loss"],
  "Customizable Losses": ["Custom Loss Function (with code template)"],
}

// Metrics data
const metricsData = {
  "Classification Metrics": [
    "torchmetrics.Accuracy()",
    "torchmetrics.Precision()",
    "torchmetrics.Recall()",
    "torchmetrics.F1Score()",
    "torchmetrics.AUROC()",
    "torchmetrics.ConfusionMatrix()",
  ],
  "Regression Metrics": [
    "torchmetrics.MeanAbsoluteError()",
    "torchmetrics.MeanSquaredError()",
    "torchmetrics.MeanSquaredLogError()",
    "torchmetrics.R2Score()",
  ],
  "Segmentation Metrics": ["torchmetrics.JaccardIndex() (IoU)", "torchmetrics.Dice()"],
  "Ranking Metrics": ["torchmetrics.RetrievalMRR()", "torchmetrics.RetrievalNormalizedDCG()"],
  "Clustering Metrics": [
    "Silhouette Score (via sklearn integration)",
    "Davies-Bouldin Index (via sklearn integration)",
  ],
  Miscellaneous: [
    "torchmetrics.CosineSimilarity()",
    "torchmetrics.ExplainedVariance()",
    "torchmetrics.SpearmanCorrcoef()",
  ],
}

export function ParametersSelection({ config, updateConfig }: ParametersSelectionProps) {
  const [monitoringCategory, setMonitoringCategory] = useState("")
  const [optimizerCategory, setOptimizerCategory] = useState("")
  const [lossCategory, setLossCategory] = useState("")
  const [metricCategory, setMetricCategory] = useState("")
  const [optimizerParams, setOptimizerParams] = useState<Record<string, string>>({})
  const [lossParams, setLossParams] = useState<Record<string, string>>({})

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
    setOptimizerParams({})
    updateConfig({
      optimizer: {
        category,
        name: "",
        params: {},
      },
    })
  }

  const handleOptimizerChange = (name: string) => {
    // Extract parameters from optimizer string
    const paramMatch = name.match(/$$(.*?)$$/)
    if (paramMatch && paramMatch[1]) {
      const params = paramMatch[1].split(",").map((p) => p.trim())
      const paramObj: Record<string, string> = {}

      params.forEach((param) => {
        const [paramName] = param.split("=")
        paramObj[paramName] = ""
      })

      setOptimizerParams(paramObj)
    } else {
      setOptimizerParams({})
    }

    updateConfig({
      optimizer: {
        category: optimizerCategory,
        name,
        params: {},
      },
    })
  }

  const handleOptimizerParamChange = (param: string, value: string) => {
    const updatedParams = { ...optimizerParams, [param]: value }
    setOptimizerParams(updatedParams)

    updateConfig({
      optimizer: {
        ...config.optimizer!,
        params: updatedParams,
      },
    })
  }

  const handleLossCategoryChange = (category: string) => {
    setLossCategory(category)
    setLossParams({})
    updateConfig({
      loss: {
        category,
        name: "",
        params: {},
      },
    })
  }

  const handleLossChange = (name: string) => {
    // Extract parameters from loss string
    const paramMatch = name.match(/$$(.*?)$$/)
    if (paramMatch && paramMatch[1]) {
      const params = paramMatch[1].split(",").map((p) => p.trim())
      const paramObj: Record<string, string> = {}

      params.forEach((param) => {
        const [paramName] = param.split("=")
        paramObj[paramName] = ""
      })

      setLossParams(paramObj)
    } else {
      setLossParams({})
    }

    updateConfig({
      loss: {
        category: lossCategory,
        name,
        params: {},
      },
    })
  }

  const handleLossParamChange = (param: string, value: string) => {
    const updatedParams = { ...lossParams, [param]: value }
    setLossParams(updatedParams)

    updateConfig({
      loss: {
        ...config.loss!,
        params: updatedParams,
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
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <h2 className="text-2xl font-bold text-slate-900">Training Parameters</h2>
        <p className="text-slate-500">Configure monitoring, optimizers, loss functions, and evaluation metrics</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
      >
        <Tabs defaultValue="monitoring" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="monitoring" className="transition-all duration-200 data-[state=active]:shadow-md">
              Monitoring
            </TabsTrigger>
            <TabsTrigger value="optimizers" className="transition-all duration-200 data-[state=active]:shadow-md">
              Optimizers
            </TabsTrigger>
            <TabsTrigger value="losses" className="transition-all duration-200 data-[state=active]:shadow-md">
              Losses
            </TabsTrigger>
            <TabsTrigger value="metrics" className="transition-all duration-200 data-[state=active]:shadow-md">
              Metrics
            </TabsTrigger>
          </TabsList>

          {/* Monitoring Tab */}
          <TabsContent value="monitoring" className="mt-4 space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <label className="text-sm font-medium text-slate-700">Category</label>
                <Select value={monitoringCategory} onValueChange={handleMonitoringCategoryChange}>
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
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
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.3 }}
              >
                <label className="text-sm font-medium text-slate-700">Option</label>
                <Select
                  value={config.monitoring?.option || ""}
                  onValueChange={handleMonitoringOptionChange}
                  disabled={!monitoringCategory}
                >
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
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
              </motion.div>
            </div>
          </TabsContent>

          {/* Optimizers Tab */}
          <TabsContent value="optimizers" className="mt-4 space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <label className="text-sm font-medium text-slate-700">Category</label>
                <Select value={optimizerCategory} onValueChange={handleOptimizerCategoryChange}>
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
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
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.3 }}
              >
                <label className="text-sm font-medium text-slate-700">Optimizer</label>
                <Select
                  value={config.optimizer?.name || ""}
                  onValueChange={handleOptimizerChange}
                  disabled={!optimizerCategory}
                >
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
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
              </motion.div>
            </div>

            {config.optimizer?.name && Object.keys(optimizerParams).length > 0 && (
              <motion.div
                className="mt-4 space-y-4 rounded-lg border border-slate-200 bg-slate-50 p-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <h3 className="text-sm font-medium text-slate-700">Optimizer Parameters</h3>
                <div className="grid gap-4 md:grid-cols-3">
                  {Object.keys(optimizerParams).map((param) => (
                    <div key={param} className="space-y-1">
                      <label className="text-xs font-medium text-slate-600">{param}</label>
                      <Input
                        value={optimizerParams[param]}
                        onChange={(e) => handleOptimizerParamChange(param, e.target.value)}
                        placeholder={`Enter ${param}`}
                        className="h-8 text-sm"
                      />
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </TabsContent>

          {/* Losses Tab */}
          <TabsContent value="losses" className="mt-4 space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <label className="text-sm font-medium text-slate-700">Category</label>
                <Select value={lossCategory} onValueChange={handleLossCategoryChange}>
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
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
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.3 }}
              >
                <label className="text-sm font-medium text-slate-700">Loss Function</label>
                <Select value={config.loss?.name || ""} onValueChange={handleLossChange} disabled={!lossCategory}>
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
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
              </motion.div>
            </div>

            {config.loss?.name && Object.keys(lossParams).length > 0 && (
              <motion.div
                className="mt-4 space-y-4 rounded-lg border border-slate-200 bg-slate-50 p-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <h3 className="text-sm font-medium text-slate-700">Loss Parameters</h3>
                <div className="grid gap-4 md:grid-cols-3">
                  {Object.keys(lossParams).map((param) => (
                    <div key={param} className="space-y-1">
                      <label className="text-xs font-medium text-slate-600">{param}</label>
                      <Input
                        value={lossParams[param]}
                        onChange={(e) => handleLossParamChange(param, e.target.value)}
                        placeholder={`Enter ${param}`}
                        className="h-8 text-sm"
                      />
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="mt-4 space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <label className="text-sm font-medium text-slate-700">Category</label>
                <Select value={metricCategory} onValueChange={handleMetricCategoryChange}>
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
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
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.3 }}
              >
                <label className="text-sm font-medium text-slate-700">Metric</label>
                <Select
                  value={config.metrics?.name || ""}
                  onValueChange={handleMetricChange}
                  disabled={!metricCategory}
                >
                  <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
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
              </motion.div>
            </div>
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  )
}
