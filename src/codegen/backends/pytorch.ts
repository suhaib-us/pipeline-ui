import { BackendGenerators } from "../types";
import { generateDataLoaderCode } from "../modules/dataLoader";
import { generateModelCode }      from "../modules/model";
import { generateOptimizerCode }  from "../modules/optimizer";
import { generateMetricsCode }    from "../modules/metrics";
import { generateTrainingCode }   from "../modules/trainingLoop";

export const PyTorchGenerators: BackendGenerators = {
  dataLoader: generateDataLoaderCode,
  model:      generateModelCode,
  optimizer:  generateOptimizerCode,
  metrics:    generateMetricsCode,
  training:   generateTrainingCode,
};