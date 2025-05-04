"use client"

import { useState, useEffect } from "react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { ModelConfig } from "./model-builder"

interface DataTypeSelectionProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

// Data type tree structure
const dataTypeTree = {
  "Image Data": ["PNG", "JPG", "BMP", "TIFF"],
  "Text Data": ["Plain Text (.txt)", "CSV", "JSON"],
  "Structured Data": ["Parquet", "CSV", "Excel (.xlsx)", "SQL Database"],
  "Audio Data": ["WAV", "MP3", "FLAC"],
  "Video Data": ["MP4", "AVI", "MOV"],
  "Medical Data": [
    "DICOM (Medical Imaging)",
    "NIfTI (Neuroimaging)",
    "EEG (Electroencephalography)",
    "Parquet",
    "HL7 (Health Level 7)",
    "FHIR (Fast Healthcare Interoperability Resources)",
  ],
  "Other Formats": ["XML", "HDF5", "Pickle"],
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
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Data Type Selection</h2>
        <p className="text-slate-500">Select the main data type and specific format</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-700">Main Data Type</label>
          <Select value={config.mainDataType} onValueChange={(value) => updateConfig({ mainDataType: value })}>
            <SelectTrigger>
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
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-700">Data Format</label>
          <Select
            value={config.subDataType}
            onValueChange={(value) => updateConfig({ subDataType: value })}
            disabled={!config.mainDataType}
          >
            <SelectTrigger>
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
        </div>
      </div>
    </div>
  )
}
