%% TEST INITIALIZATION
clear all;
clc;
name = 'DIABETES TEST';

% For reproducibility
seed = randi(100,1);
rng(seed);


testDirectory = "C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\TEST\on real dataset\TEST RESULTS\";
saveResult = 1; % if set to 1 the results of the test will be stored
[path,fid] = initTest(saveResult, name, testDirectory, seed);

%% IMPORT THE DATASET AND PREPARE IT FOR TRANING
x = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\in_diabetes.txt');
y = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\out_diabetes.txt');
x = table2array(x);
y = table2array(y);

x = zscore(x);

% Fix a missunlignment occured reading the data
y = [0;y];

% Change class values to be 1 and -1
y(y == 0) = -1;

%Shuffle the dataset
rand_pos = randperm(size(x,1)); %array of random positions
xTrainShuffle = x;
yTrainShuffle = y;
for i=1:size(x,1)
    xTrainShuffle(i,:) = x(rand_pos(i),:);
    yTrainShuffle(i,:) = y(rand_pos(i),:);
end
x = xTrainShuffle;
y = yTrainShuffle;

%% BUILDING MODELS

% Load parameters previously estimated using auto-opt function of fitcsvm
load('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Dataset analysis\var.mat', 'Cestimated', 'Sigmaestimated');

% Set parameter for the Models
C = Cestimated(1,1);
tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
tau = 1e-12;
eps = 10e-5;
maxiter = 10000;
kernel = 'gaussian';
%sigma = Sigmaestimated(1,1);
sigma = 1;

if saveResult
    fprintf(fid, 'Model parameter:\n');
    fprintf(fid, 'C = %d\n', C);
    fprintf(fid, 'tolerance = %d\n', tolerance);
    fprintf(fid, 'tau = %d\n', tau);
    fprintf(fid, 'eps = %d\n', eps);
    fprintf(fid, 'maxiter = %d\n', maxiter);
    fprintf(fid, 'kernel = %s\n', kernel);
    fprintf(fid, 'sigma used for the kernel = %s\n\n', sigma);
end

modelsNumber = 3;
models = cell(1, modelsNumber);

%Instantiate the models to be trained
FCL = FCLsmo(x, y, C, tolerance, tau, maxiter);
JSMO4 = Jsmo(x, y, C, 4, tolerance, maxiter);
JSMO6 = Jsmo(x, y, C, 6, tolerance, maxiter);

%Set the kernel
FCL.setKernel(kernel,sigma);
JSMO4.setKernel(kernel,sigma);
JSMO6.setKernel(kernel,sigma);

models{1} = FCL;
models{2} = JSMO4;
models{3} = JSMO6;

trainingStats = cell(1, size(models,2));
predictionStats = cell(1, size(models,2));
kernelEval = cell(1, size(models,2));

figureTitle{1} = "FCL version";
figureTitle{2} = 'Joachims version with q = 4';
figureTitle{3} = 'Joachims version with q = 6';

% Train the models
for k=1:size(models,2)
    
    tic
    models{k}.train();
    trainingStats{k} = toc;
    kernelEval{k} = models{k}.kernelEvaluation;
    
end

% Check the validity of the results obtained using fitcsvm
tic
fitcsvmMODEL = fitcsvm(x,y,'BoxConstraint',C,'KKTTolerance',tolerance,'IterationLimit',maxiter,'KernelFunction',kernel,'Solver','SMO');
trainingStatsFitcsvmMODEL = toc;

%% TEST THE MODELS GENERATED

% Load the test set
xt = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\in_diabetes_ts.txt');
yt = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\out_diabetes_ts.txt');
xt = table2array(xt);
yt = table2array(yt);
% Fix a missunlignment occured reading the data
yt = [0;yt];

x = zscore(x);

output = zeros(size(yt,1),size(models,2));

% Change class values to be 1 and -1
yt(yt == 0) = -1;

EvaluationStat = cell(1, size(models,2));

for k=1:size(models,2)

    tic
    output(:,k) = models{k}.predict(xt);
    predictionStats{k} = toc;
    
    % EVAL = [accuracy sensitivity specificity precision recall f_measure gmean]
    EvaluationStat{k} = Evaluate(yt,output(:,k));
    
end

tic
outputFitcsvmMODEL = fitcsvmMODEL.predict(xt);
predictionStatsFitcsvmMODEL = toc;
EvaluationStatFitcsvmModel = Evaluate(yt,outputFitcsvmMODEL);

%% SAVE TEST RESULTS

%Write Test statistics
if saveResult
    fprintf(fid, "Training statistics: \n\n");
    for k=1:size(models,2)
        fprintf(fid, compose("------------------------------------------ %s ------------------------------------------\n", figureTitle{k}));

        fprintf(fid, 'Training time %f sec\n', trainingStats{k});

        fprintf(fid, 'Prediction time %f sec\n', predictionStats{k});
        
        fprintf(fid, 'Number of iteration %d\n', models{k}.iter);
        
        fprintf(fid, 'Average iteration time %f sec\n', trainingStats{k} / models{k}.iter);
        
        fprintf(fid, 'Total number of kernel evaluation %d\n', kernelEval{k});
        
        fprintf(fid, 'Average kernel evaluation per iteration %d\n', kernelEval{k} / models{k}.iter);
        
        fprintf(fid, 'Number of support vector generated: %d\n', sum(models{k}.isSupportVector));
        
        fprintf(fid, 'Number of SV shared with fitcsvm model: %d\n', sum(and(models{k}.isSupportVector, fitcsvmMODEL.IsSupportVector) ));
        
        fprintf(fid, 'Accuracy = %f \n', EvaluationStat{k}(1,1));
        
        fprintf(fid, 'Sensitivity = %f \n', EvaluationStat{k}(1,2));
        
        fprintf(fid, 'Specificity = %f \n', EvaluationStat{k}(1,3));
        
        fprintf(fid, 'Precision = %f \n', EvaluationStat{k}(1,4));
        
        fprintf(fid, 'Recall = %f\n', EvaluationStat{k}(1,5));
        
        fprintf(fid, 'F_measure = %f \n', EvaluationStat{k}(1,6));
        
        fprintf(fid, 'Gmean = %f \n\n', EvaluationStat{k}(1,7));
        
    end
    
    fprintf(fid, "------------------------------------------ Fitcsvm ------------------------------------------\n");
    
    fprintf(fid, 'Training time %f sec\n', trainingStatsFitcsvmMODEL);
    
    fprintf(fid, 'Prediction time %f sec\n', predictionStatsFitcsvmMODEL);
    
    fprintf(fid, 'Number of support vector generated: %d\n', sum(fitcsvmMODEL.IsSupportVector));
    
    fprintf(fid, 'Accuracy = %f \n', EvaluationStatFitcsvmModel(1,1));
    
    fprintf(fid, 'Sensitivity = %f \n', EvaluationStatFitcsvmModel(1,2));
    
    fprintf(fid, 'Specificity = %f \n', EvaluationStatFitcsvmModel(1,3));
    
    fprintf(fid, 'Precision = %f \n', EvaluationStatFitcsvmModel(1,4));
    
    fprintf(fid, 'Recall = %f\n', EvaluationStatFitcsvmModel(1,5));
    
    fprintf(fid, 'F_measure = %f \n', EvaluationStatFitcsvmModel(1,6));
    
    fprintf(fid, 'Gmean = %f \n\n', EvaluationStatFitcsvmModel(1,7));
    
    fclose(fid);
end

% Save the current workspace
if saveResult
    varFile = strcat(path,'\var.mat');
    save(varFile);
end