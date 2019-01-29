%% TEST INITIALIZATION
clear all;
clc;
name = 'RINGNORM REPEAT 41 (J6)';

testDirectory = "C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\TEST\on real dataset\TEST RESULTS\";
saveResult = 1; % if set to 1 the results of the test will be stored
[path,fid] = initTest(saveResult, name, testDirectory, 4);

%% IMPORT THE TRAINING DATA
x = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\in_ringnorm.txt');
y = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\out_ringnorm.txt');
x = x(:,1:end-1);
y = y(:,1:end-1);
x = table2array(x);
y = table2array(y);
x = zscore(x);

% Change class values to be 1 and -1
y(y == 0) = -1;


%% IMPORT THE TESTING DATASET

% Load the test set
xtest = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\in_ringnorm_ts.txt');
ytest = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\out_ringnorm_ts.txt');
xtest = xtest(:,1:end-1);
ytest = ytest(:,1:end-1);
xtest = table2array(xtest);
ytest = table2array(ytest);

xtest = zscore(xtest);

% Change class values to be 1 and -1
ytest(ytest == 0) = -1;

%% COARSE GRAIN SEARCH
% A first coarse grain grid search will try several possible combination of
% parameter (C and sigma) for the model under analysis.
% The procedure will be carried out in parallel for all three methods of 
% optimization under analysis in order to collect statistics about how they
% perform on the training procedure.
% The models will be evaluate using the testing set. Altough this is not
% the correct way to procedure, as a validation set would be used instead
% of the test set. This choice is due to the fact that for the smaller
% dataset a cross-validation procedure wold be necessary and this would
% have overload the computation time.
% Furthermore the main scope of this project is to test the performances of
% the training procedures, not to create a good model for the dataset used
% during the test.

% Set parameter that are in common with all the Models
tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
tau = 1e-12;
eps = 10e-5;
maxiter = 100000;
kernel = 'gaussian';

modelsNumber = 1;
models = cell(1, modelsNumber);

% Start estimating the best sigma and C using a coarse grain grid search
C = 2^7;
sigma = 2^-15;

Rndtrial = 10;

% Instantiate arrays that will collect statistics about the training
% procedure and the grid search
trainingStats = zeros(Rndtrial, modelsNumber);
kernelEval = zeros(Rndtrial, modelsNumber);
iterNumber = zeros(Rndtrial, modelsNumber);
svNumber = zeros(Rndtrial, modelsNumber);
accuracy = cell(Rndtrial, modelsNumber+1);
violation = cell(Rndtrial, modelsNumber);

for k=1:Rndtrial

    rng(k);
    
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
    
    %Instantiate the models to be trained
    models{1} = Jsmo(x, y, C, 6, tolerance, maxiter);
    
    %Set the kernel
    models{1}.setKernel(kernel,sigma);
    
    tic
    models{1}.train();
    trainingStats(k,1) = toc;
    kernelEval(k,1) = models{1}.kernelEvaluation;
    iterNumber(k,1) = models{1}.iter;
    svNumber(k,1) = sum(models{1}.isSupportVector);
    violation{k,1} = models{1}.violation;
    fitcsvmMODEL = fitcsvm(x,y,'BoxConstraint',C,'KKTTolerance',tolerance,'IterationLimit',maxiter,'KernelFunction',kernel,'KernelScale',sigma,'Solver','SMO');
    
    output = models{1}.predict(xtest);
    accuracy{k,1} = Evaluate(ytest,output); % EVAL = [accuracy sensitivity specificity precision recall f_measure gmean]
    
    output = fitcsvmMODEL.predict(xtest);
    accuracy{k,2} = Evaluate(ytest,output); % EVAL = [accuracy sensitivity specificity precision recall f_measure gmean]
    
    % Save the variable of interest of the current workspace
    if saveResult
        varFile = strcat(path,'\gridSearch.mat');
        save(varFile,'trainingStats','kernelEval','iterNumber','svNumber','accuracy');
    end
    
end