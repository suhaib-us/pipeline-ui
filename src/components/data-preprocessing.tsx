"use client"

import { useState, useEffect } from "react"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Card, CardContent } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { ModelConfig } from "./model-builder"
import { motion } from "framer-motion"

interface DataPreprocessingProps {
  config: ModelConfig
  updateConfig: (data: Partial<ModelConfig>) => void
}

// Preprocessing options based on data type
const preprocessingOptions = {
  "Image Data": [
    "torchvision.transforms.Resize(size)",
    "torchvision.transforms.RandomCrop(size)",
    "torchvision.transforms.RandomHorizontalFlip(p)",
    "torchvision.transforms.Normalize(mean, std)",
    "torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)",
    "torchvision.transforms.ToTensor()",
    "torchvision.transforms.RandomRotation(degrees)",
    "torchvision.transforms.CenterCrop(size)",
    "torchvision.transforms.Grayscale(num_output_channels)",
  ],
  "Text Data": [
    "torchtext.transforms.ToTensor()",
    "torchtext.transforms.VocabTransform(vocab)",
    "torchtext.transforms.SentencePieceTokenizer()",
    "torchtext.transforms.Sequential",
    "torchtext.transforms.PadTransform(max_length, pad_value)",
    "torchtext.transforms.TruncateTransform(max_length)",
  ],
  "Structured Data": [
    "torch.nn.Normalize(mean, std)",
    "torch.nn.MinMaxScaler()",
    "torch.nn.StandardScaler()",
    "torch.nn.OneHotEncoder()",
    "torch.nn.MissingValueImputer(strategy)",
  ],
  "Audio Data": [
    "torchaudio.transforms.MelSpectrogram()",
    "torchaudio.transforms.MFCC()",
    "torchaudio.transforms.AmplitudeToDB()",
    "torchaudio.transforms.Resample()",
    "torchaudio.transforms.TimeStretch()",
    "torchaudio.transforms.PitchShift()",
  ],
  "Video Data": [
    "torchvision.transforms.Resize(size)",
    "torchvision.transforms.CenterCrop(size)",
    "torchvision.transforms.Normalize(mean, std)",
    "torchvision.transforms.ToTensor()",
  ],
  "Medical Data": [
    "torchvision.transforms.Resize(size)",
    "torchvision.transforms.CenterCrop(size)",
    "torchvision.transforms.Normalize(mean, std)",
    "torchvision.transforms.ToTensor()",
  ],
  "Other Formats": [
    "torch.nn.Normalize(mean, std)",
    "torch.nn.MinMaxScaler()",
    "torch.nn.StandardScaler()",
    "Custom Preprocessing Function",
  ],
}

export function DataPreprocessing({
  config,
  updateConfig,
}: DataPreprocessingProps) {
  const [availableOptions, setAvailableOptions] = useState<string[]>([])

  useEffect(() => {
    if (
      config.mainDataType &&
      preprocessingOptions[
        config.mainDataType as keyof typeof preprocessingOptions
      ]
    ) {
      setAvailableOptions(
        preprocessingOptions[
          config.mainDataType as keyof typeof preprocessingOptions
        ],
      )

      // Filter out preprocessing options that are no longer valid
      if (config.preprocessing && config.preprocessing.length > 0) {
        const validOptions = config.preprocessing.filter((option) =>
          preprocessingOptions[
            config.mainDataType as keyof typeof preprocessingOptions
          ].includes(option),
        )
        updateConfig({ preprocessing: validOptions })
      }
    } else {
      setAvailableOptions([])
      updateConfig({ preprocessing: [] })
    }
  }, [config.mainDataType])

  const handleTogglePreprocessing = (option: string) => {
    const currentPreprocessing = config.preprocessing || []

    if (currentPreprocessing.includes(option)) {
      updateConfig({
        preprocessing: currentPreprocessing.filter((item) => item !== option),
      })
    } else {
      updateConfig({
        preprocessing: [...currentPreprocessing, option],
      })
    }
  }

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h2 className="text-2xl font-bold text-slate-900">
          Data Preprocessing
        </h2>
        <p className="text-slate-500">
          Select preprocessing options for your data
        </p>
      </motion.div>

      {config.mainDataType ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Card className="border-slate-200 shadow-sm">
            <CardContent className="p-4">
              <ScrollArea className="h-[400px] pr-4">
                <div className="space-y-4">
                  {availableOptions.map((option) => (
                    <motion.div
                      key={option}
                      className="flex items-start space-x-2 rounded-md border border-slate-200 p-3 transition-all duration-200 hover:border-slate-300 hover:bg-slate-50"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2 }}
                      whileHover={{ x: 2 }}
                    >
                      <Checkbox
                        id={option}
                        checked={
                          config.preprocessing?.includes(option) || false
                        }
                        onCheckedChange={() =>
                          handleTogglePreprocessing(option)
                        }
                        className="mt-1"
                      />
                      <div className="space-y-1">
                        <Label
                          htmlFor={option}
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                        >
                          {option}
                        </Label>
                        <p className="text-xs text-slate-500">
                          {getPreprocessingDescription(option)}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </motion.div>
      ) : (
        <motion.div
          className="rounded-lg border border-dashed border-slate-300 p-8 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <p className="text-slate-500">
            Please select a data type first to see available preprocessing
            options.
          </p>
        </motion.div>
      )}
    </div>
  )
}

function getPreprocessingDescription(option: string): string {
  const descriptions: Record<string, string> = {
    "torchvision.transforms.Resize(size)":
      "Resize the input image to the given size.",
    "torchvision.transforms.RandomCrop(size)":
      "Crop the image at a random location.",
    "torchvision.transforms.RandomHorizontalFlip(p)":
      "Horizontally flip the image with probability p.",
    "torchvision.transforms.Normalize(mean, std)":
      "Normalize a tensor image with mean and standard deviation.",
    "torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)":
      "Randomly change the brightness, contrast, saturation and hue of an image.",
    "torchvision.transforms.ToTensor()":
      "Convert a PIL Image or numpy.ndarray to tensor.",
    "torchvision.transforms.RandomRotation(degrees)":
      "Rotate the image by a random angle.",
    "torchvision.transforms.CenterCrop(size)": "Crop the image at the center.",
    "torchvision.transforms.Grayscale(num_output_channels)":
      "Convert image to grayscale.",
    "torchtext.transforms.ToTensor()": "Convert text to tensor representation.",
    "torchtext.transforms.VocabTransform(vocab)":
      "Transform tokens into indices using a vocabulary.",
    "torchtext.transforms.SentencePieceTokenizer()":
      "Tokenize text using SentencePiece model.",
    "torchtext.transforms.Sequential": "Apply a sequence of transformations.",
    "torchtext.transforms.PadTransform(max_length, pad_value)":
      "Pad sequences to the same length.",
    "torchtext.transforms.TruncateTransform(max_length)":
      "Truncate sequences to a maximum length.",
    "torch.nn.Normalize(mean, std)":
      "Normalize features by mean and standard deviation.",
    "torch.nn.MinMaxScaler()": "Scale features to a given range.",
    "torch.nn.StandardScaler()":
      "Standardize features by removing the mean and scaling to unit variance.",
    "torch.nn.OneHotEncoder()":
      "Encode categorical features as a one-hot numeric array.",
    "torch.nn.MissingValueImputer(strategy)":
      "Replace missing values using a specified strategy.",
    "torchaudio.transforms.MelSpectrogram()":
      "Create a spectrogram from an audio signal.",
    "torchaudio.transforms.MFCC()":
      "Create the Mel-frequency cepstrum coefficients from an audio signal.",
    "torchaudio.transforms.AmplitudeToDB()":
      "Convert amplitude to decibel scale.",
    "torchaudio.transforms.Resample()":
      "Resample audio signal to a different frequency.",
    "torchaudio.transforms.TimeStretch()":
      "Stretch audio signal in time without changing pitch.",
    "torchaudio.transforms.PitchShift()": "Shift the pitch of an audio signal.",
    "Custom Preprocessing Function":
      "Define a custom function for preprocessing.",
  }

  return descriptions[option] || "Apply this transformation to your data."
}
