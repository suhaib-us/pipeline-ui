import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ModelConfig } from "./model-builder"
import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"

interface TrainingConfigProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

const schedulers = [
  "None",
  "torch.optim.lr_scheduler.StepLR",
  "torch.optim.lr_scheduler.MultiStepLR",
  "torch.optim.lr_scheduler.ExponentialLR",
  "torch.optim.lr_scheduler.CosineAnnealingLR",
  "torch.optim.lr_scheduler.ReduceLROnPlateau",
]

export function TrainingConfig({ config, updateConfig }: TrainingConfigProps) {
  const [schedulerParams, setSchedulerParams] = useState<Record<string, string>>({})

  const handleTrainingChange = (field: string, value: any) => {
    updateConfig({
      training: {
        ...config.training!,
        [field]: value,
      },
    })
  }

  const handleSchedulerChange = (scheduler: string) => {
    let params: Record<string, string> = {}

    if (scheduler === "torch.optim.lr_scheduler.StepLR") {
      params = { step_size: "30", gamma: "0.1" }
    } else if (scheduler === "torch.optim.lr_scheduler.MultiStepLR") {
      params = { milestones: "[30, 60, 90]", gamma: "0.1" }
    } else if (scheduler === "torch.optim.lr_scheduler.ExponentialLR") {
      params = { gamma: "0.95" }
    } else if (scheduler === "torch.optim.lr_scheduler.CosineAnnealingLR") {
      params = { T_max: "10", eta_min: "0" }
    } else if (scheduler === "torch.optim.lr_scheduler.ReduceLROnPlateau") {
      params = { mode: "min", factor: "0.1", patience: "10", threshold: "0.0001" }
    }

    setSchedulerParams(params)

    updateConfig({
      training: {
        ...config.training!,
        scheduler: scheduler === "None" ? undefined : scheduler,
        schedulerParams: scheduler === "None" ? undefined : params,
      },
    })
  }

  const handleSchedulerParamChange = (param: string, value: string) => {
    const updatedParams = { ...schedulerParams, [param]: value }
    setSchedulerParams(updatedParams)

    updateConfig({
      training: {
        ...config.training!,
        schedulerParams: updatedParams,
      },
    })
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <h2 className="text-2xl font-bold text-slate-900">Training Configuration</h2>
        <p className="text-slate-500">Configure training hyperparameters</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
      >
        <Card className="border-slate-200 shadow-sm">
          <CardContent className="p-6">
            <div className="grid gap-6 md:grid-cols-2">
              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <Label htmlFor="batch-size" className="text-sm font-medium text-slate-700">
                  Batch Size
                </Label>
                <Input
                  id="batch-size"
                  type="number"
                  value={config.training?.batchSize || 32}
                  onChange={(e) => handleTrainingChange("batchSize", Number.parseInt(e.target.value))}
                  className="transition-all duration-200 hover:border-slate-400"
                />
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.3 }}
              >
                <Label htmlFor="epochs" className="text-sm font-medium text-slate-700">
                  Number of Epochs
                </Label>
                <Input
                  id="epochs"
                  type="number"
                  value={config.training?.epochs || 10}
                  onChange={(e) => handleTrainingChange("epochs", Number.parseInt(e.target.value))}
                  className="transition-all duration-200 hover:border-slate-400"
                />
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.4 }}
              >
                <Label htmlFor="learning-rate" className="text-sm font-medium text-slate-700">
                  Learning Rate
                </Label>
                <Input
                  id="learning-rate"
                  type="number"
                  step="0.0001"
                  value={config.training?.learningRate || 0.001}
                  onChange={(e) => handleTrainingChange("learningRate", Number.parseFloat(e.target.value))}
                  className="transition-all duration-200 hover:border-slate-400"
                />
              </motion.div>

              <motion.div
                className="space-y-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.5 }}
              >
                <Label htmlFor="weight-decay" className="text-sm font-medium text-slate-700">
                  Weight Decay
                </Label>
                <Input
                  id="weight-decay"
                  type="number"
                  step="0.0001"
                  value={config.training?.weightDecay || 0.0001}
                  onChange={(e) => handleTrainingChange("weightDecay", Number.parseFloat(e.target.value))}
                  className="transition-all duration-200 hover:border-slate-400"
                />
              </motion.div>
            </div>

            <motion.div
              className="mt-6 flex items-start space-x-2"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.6 }}
            >
              <Checkbox
                id="early-stopping"
                checked={config.training?.earlyStoppingEnabled || false}
                onCheckedChange={(checked) => handleTrainingChange("earlyStoppingEnabled", checked)}
                className="mt-1"
              />
              <div className="space-y-1">
                <Label
                  htmlFor="early-stopping"
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  Enable Early Stopping
                </Label>
                <p className="text-xs text-slate-500">Stop training when a monitored metric has stopped improving.</p>
              </div>
            </motion.div>

            {config.training?.earlyStoppingEnabled && (
              <motion.div
                className="mt-4 grid gap-6 rounded-lg border border-slate-200 bg-slate-50 p-4 md:grid-cols-2"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                transition={{ duration: 0.3 }}
              >
                <div className="space-y-2">
                  <Label htmlFor="patience" className="text-sm font-medium text-slate-700">
                    Patience
                  </Label>
                  <Input
                    id="patience"
                    type="number"
                    value={config.training?.earlyStoppingPatience || 5}
                    onChange={(e) => handleTrainingChange("earlyStoppingPatience", Number.parseInt(e.target.value))}
                    className="transition-all duration-200 hover:border-slate-400"
                  />
                  <p className="text-xs text-slate-500">
                    Number of epochs with no improvement after which training will be stopped.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="min-delta" className="text-sm font-medium text-slate-700">
                    Min Delta
                  </Label>
                  <Input
                    id="min-delta"
                    type="number"
                    step="0.0001"
                    value={config.training?.earlyStoppingMinDelta || 0.0001}
                    onChange={(e) => handleTrainingChange("earlyStoppingMinDelta", Number.parseFloat(e.target.value))}
                    className="transition-all duration-200 hover:border-slate-400"
                  />
                  <p className="text-xs text-slate-500">
                    Minimum change in the monitored quantity to qualify as an improvement.
                  </p>
                </div>
              </motion.div>
            )}

            <motion.div
              className="mt-6 space-y-2"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.7 }}
            >
              <Label htmlFor="scheduler" className="text-sm font-medium text-slate-700">
                Learning Rate Scheduler
              </Label>
              <Select value={config.training?.scheduler || "None"} onValueChange={handleSchedulerChange}>
                <SelectTrigger id="scheduler" className="transition-all duration-200 hover:border-slate-400">
                  <SelectValue placeholder="Select a scheduler" />
                </SelectTrigger>
                <SelectContent>
                  {schedulers.map((scheduler) => (
                    <SelectItem key={scheduler} value={scheduler}>
                      {scheduler}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </motion.div>

            {config.training?.scheduler && config.training.scheduler !== "None" && (
              <motion.div
                className="mt-4 space-y-4 rounded-lg border border-slate-200 bg-slate-50 p-4"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                transition={{ duration: 0.3 }}
              >
                <h3 className="text-sm font-medium text-slate-700">Scheduler Parameters</h3>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {Object.keys(schedulerParams).map((param) => (
                    <div key={param} className="space-y-1">
                      <Label htmlFor={param} className="text-xs font-medium text-slate-600">
                        {param}
                      </Label>
                      <Input
                        id={param}
                        value={schedulerParams[param]}
                        onChange={(e) => handleSchedulerParamChange(param, e.target.value)}
                        className="h-8 text-sm transition-all duration-200 hover:border-slate-400"
                      />
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
