import { useState, useEffect } from "react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ModelConfig } from "./model-builder"

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
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Task Selection</h2>
        <p className="text-slate-500">Select the main task category and specific sub-task</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-700">Main Task Category</label>
          <Select value={config.mainTask} onValueChange={(value) => updateConfig({ mainTask: value })}>
            <SelectTrigger>
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
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-700">Sub-task</label>
          <Select
            value={config.subTask}
            onValueChange={(value) => updateConfig({ subTask: value })}
            disabled={!config.mainTask}
          >
            <SelectTrigger>
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
        </div>
      </div>
    </div>
  )
}
