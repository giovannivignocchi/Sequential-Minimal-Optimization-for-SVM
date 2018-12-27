function [models,figureTitle] = initModel(xTrain, yTrain, C, q, tolerance, eps, tau, maxiter, kernel)

modelsNumber = 5;
models = cell(1, modelsNumber);
figureTitle = cell(1, modelsNumber);

PlattSmo = smo(xTrain, yTrain, C, tolerance, eps, maxiter);
PlattSmo.setKernel(kernel);
errorSmo = smoErrorCache(xTrain, yTrain, C, tolerance, eps, maxiter);
errorSmo.setKernel(kernel);
FenChinLinSmo = FCLsmo(xTrain, yTrain, C, tolerance, tau, maxiter);
FenChinLinSmo.setKernel(kernel);
Keerthi = KeerthiSmo(xTrain, yTrain, C, tolerance, eps, maxiter);
Keerthi.setKernel(kernel);
Joachims = Jsmo(xTrain, yTrain, C, q, tolerance, maxiter);
Joachims.setKernel(kernel);

models{1} = PlattSmo;
models{2} = errorSmo;
models{3} = FenChinLinSmo;
models{4} = Keerthi;
models{5} = Joachims;

figureTitle{1} = "PLATT version";
figureTitle{2} = "PLATT version with Error cache";
figureTitle{3} = "Fan Chen and Lin version";
figureTitle{4} = "Keerthi version";
testName = compose("Joachims version with q = %d", q);
figureTitle{5} = testName;
end

