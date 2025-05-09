import { ModelBuilder } from "@/components/model-builder"

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="mx-auto max-w-6xl">
        <h1 className="mb-6 text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">PyTorch Model Builder</h1>
        <p className="mb-8 text-slate-600">
          Build and configure your PyTorch model in a few simple steps, then download the generated code.
        </p>
        <ModelBuilder />
      </div>
    </main>
  )
}
