import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ModeSelection } from "@/components/mode-selection"
import { TaskSelection } from "@/components/task-selection"
import { DataTypeSelection } from "@/components/data-type-selection"
import { ModelSelection } from "@/components/model-selection"
import { ParametersSelection } from "@/components/parameters-selection"
import { Inference } from "@/components/inference"
import { StepIndicator } from "@/components/step-indicator"

export type ModelConfig = {
  mode: string
  mainTask: string
  subTask: string
  mainDataType: string
  subDataType: string
  modelType: "pretrained" | "custom"
  pretrainedModelCategory?: string
  pretrainedModel?: string
  customLayers?: Array<{
    type: string
    category: string
    config: Record<string, any>
  }>
  monitoring?: {
    category: string
    option: string
  }
  optimizer?: {
    category: string
    name: string
  }
  loss?: {
    category: string
    name: string
  }
  metrics?: {
    category: string
    name: string
  }
}

export function ModelBuilder() {
  const [currentStep, setCurrentStep] = useState(1)
  const [config, setConfig] = useState<ModelConfig>({
    mode: "",
    mainTask: "",
    subTask: "",
    mainDataType: "",
    subDataType: "",
    modelType: "pretrained",
    customLayers: [],
  })

  const steps = ["Mode Selection", "Task Selection", "Data Type", "Model Selection", "Parameters", "Inference"]

  const updateConfig = (newData: Partial<ModelConfig>) => {
    setConfig((prev) => ({ ...prev, ...newData }))
  }

  const handleNext = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return <ModeSelection config={config} updateConfig={updateConfig} />
      case 2:
        return <TaskSelection config={config} updateConfig={updateConfig} />
      case 3:
        return <DataTypeSelection config={config} updateConfig={updateConfig} />
      case 4:
        return <ModelSelection config={config} updateConfig={updateConfig} />
      case 5:
        return <ParametersSelection config={config} updateConfig={updateConfig} />
      case 6:
        return <Inference config={config} />
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      <StepIndicator steps={steps} currentStep={currentStep} />

      <Card className="border-slate-200 shadow-sm">
        <CardContent className="p-6">{renderStep()}</CardContent>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" onClick={handleBack} disabled={currentStep === 1}>
          Back
        </Button>
        <Button onClick={handleNext} disabled={currentStep === steps.length}>
          {currentStep === steps.length - 1 ? "Finish" : "Next"}
        </Button>
      </div>
    </div>
  )
}
