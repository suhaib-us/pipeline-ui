import { useEffect, useState } from "react"
import type { ModelConfig } from "./model-builder"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Check, Copy } from "lucide-react"

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
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Inference</h2>
        <p className="text-slate-500">Review your configuration and submit</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Configuration Summary</CardTitle>
          <CardDescription>This is the complete configuration that will be used for your model.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <Button variant="outline" size="sm" className="absolute right-2 top-2" onClick={copyToClipboard}>
              {copied ? (
                <>
                  <Check className="mr-2 h-4 w-4" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="mr-2 h-4 w-4" />
                  Copy
                </>
              )}
            </Button>
            <ScrollArea className="h-[400px] rounded-md border">
              <pre className="p-4 text-sm">{jsonOutput}</pre>
            </ScrollArea>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSubmit}>Submit Configuration</Button>
      </div>
    </div>
  )
}
