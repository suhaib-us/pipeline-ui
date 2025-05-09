import { useState, useEffect } from "react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ModelConfig } from "./model-builder"
import { motion } from "framer-motion"

interface DataTypeSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

// Data type tree structure
const dataTypeTree = {
  "Image Data": ["PNG", "JPG", "BMP", "TIFF", "PyTorch Tensor Format"],
  "Text Data": ["Plain Text (.txt)", "CSV", "JSON", "PyTorch Text Dataset Format"],
  "Structured Data": ["Parquet", "CSV", "Excel (.xlsx)", "SQL Database", "PyTorch Tabular Dataset"],
  "Audio Data": ["WAV", "MP3", "FLAC", "PyTorch Audio Format"],
  "Video Data": ["MP4", "AVI", "MOV", "PyTorch Video Format"],
  "Medical Data": [
    "DICOM (Medical Imaging)",
    "NIfTI (Neuroimaging)",
    "EEG (Electroencephalography)",
    "HL7 (Health Level 7)",
    "FHIR (Fast Healthcare Interoperability Resources)",
  ],
  "Other Formats": ["XML", "HDF5", "Pickle", "Custom PyTorch Dataset"],
}

export function DataTypeSelection({ config, updateConfig }: DataTypeSelectionProps) {
  const [subTypes, setSubTypes] = useState<string[]>([])

  useEffect(() => {
    if (config.mainDataType && dataTypeTree[config.mainDataType as keyof typeof dataTypeTree]) {
      setSubTypes(dataTypeTree[config.mainDataType as keyof typeof dataTypeTree])

      // Reset subtype if the main data type changes
      if (!dataTypeTree[config.mainDataType as keyof typeof dataTypeTree].includes(config.subDataType)) {
        updateConfig({ subDataType: "" })
      }
    } else {
      setSubTypes([])
    }
  }, [config.mainDataType])

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
        <h2 className="text-2xl font-bold text-slate-900">Data Type Selection</h2>
        <p className="text-slate-500">Select the main data type and specific format</p>
      </motion.div>

      <div className="grid gap-6 md:grid-cols-2">
        <motion.div
          className="space-y-2"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <label className="text-sm font-medium text-slate-700">Main Data Type</label>
          <Select value={config.mainDataType} onValueChange={(value) => updateConfig({ mainDataType: value })}>
            <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
              <SelectValue placeholder="Select a data type" />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(dataTypeTree).map((type) => (
                <SelectItem key={type} value={type}>
                  {type}
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
          <label className="text-sm font-medium text-slate-700">Data Format</label>
          <Select
            value={config.subDataType}
            onValueChange={(value) => updateConfig({ subDataType: value })}
            disabled={!config.mainDataType}
          >
            <SelectTrigger className="transition-all duration-200 hover:border-slate-400">
              <SelectValue placeholder="Select a data format" />
            </SelectTrigger>
            <SelectContent>
              {subTypes.map((type) => (
                <SelectItem key={type} value={type}>
                  {type}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </motion.div>
      </div>
    </div>
  )
}
