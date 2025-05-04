"use client"

import { useEffect, useState } from "react"
import type { ModelConfig } from "./model-builder"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Check, Copy, Send } from "lucide-react"
import { motion } from "framer-motion"

interface InferenceProps {
  config: ModelConfig
}

export function Inference({ config }: InferenceProps) {
  const [jsonOutput, setJsonOutput] = useState("")
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    // Format the config as a pretty JSON string
    const formattedJson = JSON.stringify(config, null, 2)
    setJsonOutput(formattedJson)
  }, [config])

  const copyToClipboard = () => {
    navigator.clipboard.writeText(jsonOutput)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleSubmit = () => {
    // This would typically send the configuration to an API endpoint
    console.log("Submitting configuration:", config)
    alert("Configuration submitted successfully!")
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <h2 className="text-2xl font-bold text-slate-900">Inference</h2>
        <p className="text-slate-500">Review your configuration and submit</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        whileHover={{ y: -2 }}
      >
        <Card className="border-slate-200 shadow-sm transition-all duration-300 hover:shadow-md">
          <CardHeader>
            <CardTitle>Configuration Summary</CardTitle>
            <CardDescription>This is the complete configuration that will be used for your model.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative">
              <Button
                variant="outline"
                size="sm"
                className={`absolute right-2 top-2 transition-all duration-200 ${
                  copied ? "bg-green-50 text-green-600" : "hover:bg-slate-100"
                }`}
                onClick={copyToClipboard}
              >
                {copied ? (
                  <motion.div
                    className="flex items-center"
                    initial={{ scale: 0.8 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 500, damping: 15 }}
                  >
                    <Check className="mr-2 h-4 w-4" />
                    Copied
                  </motion.div>
                ) : (
                  <motion.div className="flex items-center">
                    <Copy className="mr-2 h-4 w-4" />
                    Copy
                  </motion.div>
                )}
              </Button>
              <ScrollArea className="h-[400px] rounded-md border">
                <pre className="p-4 text-sm">{jsonOutput}</pre>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        className="flex justify-end"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3, delay: 0.3 }}
      >
        <Button onClick={handleSubmit} className="gap-2 transition-all duration-200 hover:scale-105">
          <Send className="h-4 w-4" />
          Submit Configuration
        </Button>
      </motion.div>
    </div>
  )
}
