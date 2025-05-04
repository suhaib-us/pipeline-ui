"use client"

import { CardDescription, CardTitle } from "@/components/ui/card"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import type { ModelConfig } from "./model-builder"

interface ModeSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

export function ModeSelection({ config, updateConfig }: ModeSelectionProps) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Mode Selection</h2>
        <p className="text-slate-500">Choose between Deep Learning or Machine Learning</p>
      </div>

      <RadioGroup
        value={config.mode}
        onValueChange={(value) => updateConfig({ mode: value })}
        className="grid grid-cols-1 gap-4 md:grid-cols-2"
      >
        <div>
          <RadioGroupItem value="dl" id="dl" className="peer sr-only" />
          <Label
            htmlFor="dl"
            className="flex cursor-pointer flex-col rounded-lg border border-slate-200 bg-white p-6 shadow-sm hover:border-slate-300 peer-data-[state=checked]:border-primary peer-data-[state=checked]:ring-1 peer-data-[state=checked]:ring-primary"
          >
            <CardTitle className="text-xl">Deep Learning</CardTitle>
            <CardDescription className="mt-2">
              Neural networks with multiple layers for complex pattern recognition and feature learning.
            </CardDescription>
          </Label>
        </div>

        <div>
          <RadioGroupItem value="ml" id="ml" className="peer sr-only" />
          <Label
            htmlFor="ml"
            className="flex cursor-pointer flex-col rounded-lg border border-slate-200 bg-white p-6 shadow-sm hover:border-slate-300 peer-data-[state=checked]:border-primary peer-data-[state=checked]:ring-1 peer-data-[state=checked]:ring-primary"
          >
            <CardTitle className="text-xl">Machine Learning</CardTitle>
            <CardDescription className="mt-2">
              Traditional algorithms that learn patterns from data without deep neural architectures.
            </CardDescription>
          </Label>
        </div>
      </RadioGroup>
    </div>
  )
}
