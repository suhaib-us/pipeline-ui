"use client"

import { useState, useEffect } from "react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ModelConfig } from "./model-builder"
import { motion } from "framer-motion"

interface TaskSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

// Task tree data structure
const dlTaskTree = {
  "Image Processing": [
    "Image Classification",
    "Object Detection",
    "Image Segmentation",
    "Image Generation",
    "Style Transfer",
  ],
  "Text Processing": [
    "Text Classification",
    "Sentiment Analysis",
    "Named Entity Recognition",
    "Text Generation",
    "Machine Translation",
    "Text Summarization",
  ],
  "Audio Processing": ["Speech Recognition", "Audio Classification", "Audio Generation", "Voice Conversion"],
  "Generative Models": [
    "GAN (Generative Adversarial Networks)",
    "VAE (Variational Autoencoders)",
    "Diffusion Models",
    "Autoregressive Models",
  ],
  "Reinforcement Learning": [
    "Value-based Methods (DQN, etc.)",
    "Policy-based Methods (PPO, etc.)",
    "Model-based Methods",
    "Multi-agent Systems",
  ],
  "Other Tasks": ["Anomaly Detection", "Dimensionality Reduction", "Recommendation Systems", "Time Series Analysis"],
}

const mlTaskTree = {
  Classification: ["Binary Classification", "Multi-class Classification", "Multi-label Classification"],
  Regression: ["Simple Regression", "Multiple Regression", "Polynomial Regression", "Time Series Forecasting"],
  Clustering: ["K-Means", "Hierarchical Clustering", "DBSCAN", "Gaussian Mixture Models"],
  "Dimensionality Reduction": ["PCA (Principal Component Analysis)", "t-SNE", "UMAP", "Autoencoders"],
  "Anomaly Detection": ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Autoencoder-based Detection"],
  "Recommendation Systems": [
    "Collaborative Filtering",
    "Content-Based Filtering",
    "Hybrid Systems",
    "Matrix Factorization",
  ],
}

export function TaskSelection({ config, updateConfig }: TaskSelectionProps) {
  const [mainTasks, setMainTasks] = useState<string[]>([])
  const [subTasks, setSubTasks] = useState<string[]>([])

  useEffect(() => {
    if (config.mode === "dl") {
      setMainTasks(Object.keys(dlTaskTree))
    } else if (config.mode === "ml") {
      setMainTasks(Object.keys(mlTaskTree))
    } else {
      setMainTasks([])
    }

    // Reset main task if mode changes
    if (
      config.mode &&
      ((config.mode === "dl" && !Object.keys(dlTaskTree).includes(config.mainTask)) ||
        (config.mode === "ml" && !Object.keys(mlTaskTree).includes(config.mainTask)))
    ) {
      updateConfig({ mainTask: "", subTask: "" })
    }
  }, [config.mode])

  useEffect(() => {
    if (config.mainTask) {
      if (config.mode === "dl" && dlTaskTree[config.mainTask as keyof typeof dlTaskTree]) {
        setSubTasks(dlTaskTree[config.mainTask as keyof typeof dlTaskTree])
      } else if (config.mode === "ml" && mlTaskTree[config.mainTask as keyof typeof mlTaskTree]) {
        setSubTasks(mlTaskTree[config.mainTask as keyof typeof mlTaskTree])
      } else {
        setSubTasks([])
      }

      // Reset subtask if main task changes
      const taskTree = config.mode === "dl" ? dlTaskTree : mlTaskTree;
      if (config.mainTask && taskTree.hasOwnProperty(config.mainTask)) {
        const selectedTaskTree = taskTree[config.mainTask as keyof typeof taskTree] as string[];
        if (selectedTaskTree && !selectedTaskTree.includes(config.subTask)) {
          updateConfig({ subTask: "" });
        }
      }
    } else {
      setSubTasks([]);
    }
  }, [config.mainTask, config.mode]);

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <h2 className="text-2xl font-bold text-slate-900">Task Selection</h2>
        <p className="text-slate-500">Select the main task category and specific sub-task</p>
      </motion.div>

      <div className="grid gap-6 md:grid-cols-2">
        <motion.div
          className="space-y-2"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <label className="text-sm font-medium text-slate-700">Main Task Category</label>
          <Select
            value={config.mainTask}
            onValueChange={(value) => updateConfig({ mainTask: value })}
            disabled={!config.mode}
          >
            <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
              <SelectValue placeholder="Select a task category" />
            </SelectTrigger>
            <SelectContent>
              {mainTasks.map((task) => (
                <SelectItem key={task} value={task}>
                  {task}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </motion.div>

        <motion.div
          className="space-y-2"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <label className="text-sm font-medium text-slate-700">Sub-task</label>
          <Select
            value={config.subTask}
            onValueChange={(value) => updateConfig({ subTask: value })}
            disabled={!config.mainTask}
          >
            <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
              <SelectValue placeholder="Select a sub-task" />
            </SelectTrigger>
            <SelectContent>
              {subTasks.map((task) => (
                <SelectItem key={task} value={task}>
                  {task}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </motion.div>
      </div>
    </div>
  )
}
