"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ModeSelection } from "@/components/mode-selection"
import { TaskSelection } from "@/components/task-selection"
import { DataTypeSelection } from "@/components/data-type-selection"
import { DataPreprocessing } from "@/components/data-preprocessing"
import { ModelSelection } from "@/components/model-selection"
import { ParametersSelection } from "@/components/parameters-selection"
import { TrainingConfig } from "@/components/training-config"
import { CodeGeneration } from "@/components/code-generation" 
import { StepIndicator } from "@/components/step-indicator"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowLeft, ArrowRight } from "lucide-react"

export type ModelConfig = {
  mode: string
  mainTask: string
  subTask: string
  mainDataType: string
  subDataType: string
  preprocessing: string[]
  fileMetadata: any
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
    params?: Record<string, any>
  }
  loss?: {
    category: string
    name: string
    params?: Record<string, any>
  }
  metrics?: {
    category: string
    name: string
  }
  training?: {
    batchSize: number
    epochs: number
    learningRate: number
    weightDecay: number
    earlyStoppingEnabled: boolean
    earlyStoppingPatience?: number
    earlyStoppingMinDelta?: number
    scheduler?: string
    schedulerParams?: Record<string, any>
  }
}

export function ModelBuilder() {
  const [currentStep, setCurrentStep] = useState(1)
  const [direction, setDirection] = useState(0)
  const [config, setConfig] = useState<ModelConfig>({
    mode: "",
    mainTask: "",
    subTask: "",
    mainDataType: "",
    subDataType: "",
    preprocessing: [],
    fileMetadata: null,
    modelType: "pretrained",
    customLayers: [],
    training: {
      batchSize: 32,
      epochs: 10,
      learningRate: 0.001,
      weightDecay: 0.0001,
      earlyStoppingEnabled: false,
    },
  })

  const steps = [
    "Framework Selection",
    "Task Selection",
    "Data Type",
    "Data Preprocessing",
    "Model Selection",
    "Training Parameters",
    "Training Configuration",
    "Code Generation",
  ]

  const updateConfig = (newData: Partial<ModelConfig>) => {
    setConfig((prev) => ({ ...prev, ...newData }))
  }

  const handleNext = () => {
    if (currentStep < steps.length) {
      setDirection(1)
      setCurrentStep(currentStep + 1)
      window.scrollTo({ top: 0, behavior: "smooth" })
    }
  }

  const handleBack = () => {
    if (currentStep > 1) {
      setDirection(-1)
      setCurrentStep(currentStep - 1)
      window.scrollTo({ top: 0, behavior: "smooth" })
    }
  }

  const variants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 50 : -50,
      opacity: 0,
    }),
    center: {
      x: 0,
      opacity: 1,
    },
    exit: (direction: number) => ({
      x: direction < 0 ? 50 : -50,
      opacity: 0,
    }),
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
        return <DataPreprocessing config={config} updateConfig={updateConfig} />
      case 5:
        return <ModelSelection config={config} updateConfig={updateConfig} />
      case 6:
        return <ParametersSelection config={config} updateConfig={updateConfig} />
      case 7:
        return <TrainingConfig config={config} updateConfig={updateConfig} />
      case 8:
        return <CodeGeneration config={config} />
      default:
        return null
    }
  }

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <StepIndicator steps={steps} currentStep={currentStep} />

      <Card className="border-slate-200 shadow-sm overflow-hidden">
        <CardContent className="p-6">
          <AnimatePresence mode="wait" custom={direction}>
            <motion.div
              key={currentStep}
              custom={direction}
              variants={variants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{
                x: { type: "spring", stiffness: 300, damping: 30 },
                opacity: { duration: 0.2 },
              }}
            >
              {renderStep()}
            </motion.div>
          </AnimatePresence>
        </CardContent>
      </Card>

      <motion.div
        className="flex justify-between"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <Button
          variant="outline"
          onClick={handleBack}
          disabled={currentStep === 1}
          className="gap-2 transition-all duration-200 hover:translate-x-[-2px]"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>
        <Button
          onClick={handleNext}
          disabled={currentStep === steps.length}
          className="gap-2 transition-all duration-200 hover:translate-x-[2px]"
        >
          {currentStep === steps.length - 1 ? "Generate Code" : "Next"}
          {currentStep !== steps.length && <ArrowRight className="h-4 w-4" />}
        </Button>
      </motion.div>
    </motion.div>
  )
}
