import { useEffect, useState } from "react"
import type { ModelConfig } from "./model-builder"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Check, Copy, Code } from "lucide-react"
import { motion } from "framer-motion"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CodeGeneration } from "./code-generation"

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

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <h2 className="text-2xl font-bold text-slate-900">Inference & Code Generation</h2>
        <p className="text-slate-500">Review your configuration and generate PyTorch code</p>
      </motion.div>

      <Tabs defaultValue="code" className="w-full">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="code" className="transition-all duration-200 data-[state=active]:shadow-md">
              <Code className="mr-2 h-4 w-4" />
              Generated Code
            </TabsTrigger>
            <TabsTrigger value="config" className="transition-all duration-200 data-[state=active]:shadow-md">
              Configuration JSON
            </TabsTrigger>
          </TabsList>
        </motion.div>

        <TabsContent value="code" className="mt-4">
          <CodeGeneration config={config} />
        </TabsContent>

        <TabsContent value="config" className="mt-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            whileHover={{ y: -2 }}
          >
            <Card className="border-slate-200 shadow-sm transition-all duration-300 hover:shadow-md">
              <CardHeader>
                <CardTitle>Configuration JSON</CardTitle>
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
        </TabsContent>
      </Tabs>
    </div>
  )
}
