import { CheckIcon } from "lucide-react"
import { motion } from "framer-motion"

interface StepIndicatorProps {
  steps: string[]
  currentStep: number
}

export function StepIndicator({ steps, currentStep }: StepIndicatorProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const stepNumber = index + 1
          const isActive = stepNumber === currentStep
          const isCompleted = stepNumber < currentStep

          return (
            <div key={step} className="flex flex-col items-center">
              <motion.div
                className={`flex h-10 w-10 items-center justify-center rounded-full border-2 ${
                  isActive
                    ? "border-primary bg-primary text-primary-foreground"
                    : isCompleted
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-slate-300 bg-slate-50 text-slate-500"
                }`}
                initial={false}
                animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                transition={{ duration: 0.5 }}
              >
                {isCompleted ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <CheckIcon className="h-5 w-5" />
                  </motion.div>
                ) : (
                  <span>{stepNumber}</span>
                )}
              </motion.div>
              <motion.span
                className={`mt-2 text-xs ${isActive || isCompleted ? "font-medium text-slate-900" : "text-slate-500"}`}
                animate={isActive ? { fontWeight: 600 } : {}}
              >
                {step}
              </motion.span>
            </div>
          )
        })}
      </div>

      <div className="relative mt-4">
        <div className="absolute left-0 top-1/2 h-0.5 w-full -translate-y-1/2 bg-slate-200"></div>
        <motion.div
          className="absolute left-0 top-1/2 h-0.5 -translate-y-1/2 bg-primary"
          initial={false}
          animate={{ width: `${((currentStep - 1) / (steps.length - 1)) * 100}%` }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
        ></motion.div>
      </div>
    </div>
  )
}
