import { CardDescription, CardTitle } from "@/components/ui/card"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import type { ModelConfig } from "./model-builder"
import { motion } from "framer-motion"

interface ModeSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

export function ModeSelection({ config, updateConfig }: ModeSelectionProps) {
  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <h2 className="text-2xl font-bold text-slate-900">Mode Selection</h2>
        <p className="text-slate-500">Choose between Deep Learning or Machine Learning</p>
      </motion.div>

      <RadioGroup
        value={config.mode}
        onValueChange={(value) => updateConfig({ mode: value })}
        className="grid grid-cols-1 gap-4 md:grid-cols-2"
      >
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <RadioGroupItem value="dl" id="dl" className="peer sr-only" />
          <Label
            htmlFor="dl"
            className="flex cursor-pointer flex-col rounded-lg border border-slate-200 bg-white p-6 shadow-sm transition-all duration-200 hover:border-slate-300 hover:shadow-md peer-data-[state=checked]:border-primary peer-data-[state=checked]:ring-1 peer-data-[state=checked]:ring-primary"
          >
            <CardTitle className="text-xl">Deep Learning</CardTitle>
            <CardDescription className="mt-2">
              Neural networks with multiple layers for complex pattern recognition and feature learning.
            </CardDescription>
          </Label>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <RadioGroupItem value="ml" id="ml" className="peer sr-only" />
          <Label
            htmlFor="ml"
            className="flex cursor-pointer flex-col rounded-lg border border-slate-200 bg-white p-6 shadow-sm transition-all duration-200 hover:border-slate-300 hover:shadow-md peer-data-[state=checked]:border-primary peer-data-[state=checked]:ring-1 peer-data-[state=checked]:ring-primary"
          >
            <CardTitle className="text-xl">Machine Learning</CardTitle>
            <CardDescription className="mt-2">
              Traditional algorithms that learn patterns from data without deep neural architectures.
            </CardDescription>
          </Label>
        </motion.div>
      </RadioGroup>
    </div>
  )
}
