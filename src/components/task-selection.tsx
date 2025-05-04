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
const taskTree = {
  "Image Processing": ["Image Classification", "Object Detection", "Image Segmentation"],
  "Text Processing": ["Text Classification", "Sentiment Analysis", "Named Entity Recognition", "Text Generation"],
  "Traditional Machine Learning": ["Classification", "Regression", "Clustering"],
  "Reinforcement Learning": ["Value-based Methods", "Policy-based Methods", "Model-based Methods"],
  "Other Tasks": ["Anomaly Detection", "Dimensionality Reduction", "Recommendation Systems"],
}

export function TaskSelection({ config, updateConfig }: TaskSelectionProps) {
  const [subTasks, setSubTasks] = useState<string[]>([])

  useEffect(() => {
    if (config.mainTask && taskTree[config.mainTask as keyof typeof taskTree]) {
      setSubTasks(taskTree[config.mainTask as keyof typeof taskTree])

      // Reset subtask if the main task changes
      if (!taskTree[config.mainTask as keyof typeof taskTree].includes(config.subTask)) {
        updateConfig({ subTask: "" })
      }
    } else {
      setSubTasks([])
    }
  }, [config.mainTask])

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
          <Select value={config.mainTask} onValueChange={(value) => updateConfig({ mainTask: value })}>
            <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
              <SelectValue placeholder="Select a task category" />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(taskTree).map((task) => (
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
