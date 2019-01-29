%% TEST INITIALIZATION
clear all;
clc;
name = 'DIABETES';

% For reproducibility
%seed = randi(100,1);
seed = 66;
rng(seed);


testDirectory = "C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\TEST\on real dataset\TEST RESULTS\";
saveResult = 1; % if set to 1 the results of the test will be stored
[path,fid] = initTest(saveResult, name, testDirectory, seed);

%% IMPORT THE TRAINING DATA
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

%% IMPORT THE TESTING DATASET

% Load the test set
xtets = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\in_diabetes_ts.txt');
ytest = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\out_diabetes_ts.txt');
xtets = table2array(xtets);
ytest = table2array(ytest);
% Fix a missunlignment occured reading the data
ytest = [0;ytest];

xtets = zscore(xtets);

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

modelsNumber = 3;
models = cell(1, modelsNumber);

% Start estimating the best sigma and C using a coarse grain grid search
C = [2^-5, 2^-3, 2, 2^7, 2^9, 2^15]; % 6
sigma = [2^-15, 2^-13, 2^-11, 2^-9, 2^-7, 2^-5, 2^-3, 2^-1, 2, 2^3]; % 10

GridSize = size(C,2) * size(sigma,2);

% Instantiate arrays that will collect statistics about the training
% procedure and the grid search
trainingStats = zeros(GridSize, modelsNumber);
kernelEval = zeros(GridSize, modelsNumber);
iterNumber = zeros(GridSize, modelsNumber);
svNumber = zeros(GridSize, modelsNumber);
violation = cell(GridSize, modelsNumber);
accuracy = cell(GridSize, modelsNumber+1);

modelParam = zeros(GridSize, 2);
iter = 0;

for c=C
    for s = sigma

        iter = iter + 1;
        modelParam(iter,1) = c;
        modelParam(iter,2) = s;
        
        %Instantiate the models to be trained
        models{1} = FCLsmo(x, y, c, tolerance, tau, maxiter);
        models{2} = Jsmo(x, y, c, 4, tolerance, maxiter);
        models{3} = Jsmo(x, y, c, 6, tolerance, maxiter);

        %Set the kernel
        models{1}.setKernel(kernel,s);
        models{2}.setKernel(kernel,s);
        models{3}.setKernel(kernel,s);
        
        % Train the models
        for k=1:modelsNumber
    
            tic
            models{k}.train();
            trainingStats(iter,k) = toc;
            kernelEval(iter,k) = models{k}.kernelEvaluation;
            iterNumber(iter,k) = models{k}.iter;
            svNumber(iter,k) = sum(models{k}.isSupportVector);
            violation{iter,k} = models{k}.violation;
            
        end
        
        fitcsvmMODEL = fitcsvm(x,y,'BoxConstraint',c,'KKTTolerance',tolerance,'IterationLimit',maxiter,'KernelFunction',kernel,'KernelScale',s,'Solver','SMO');
        
        %Evaluate the model generated on the test set
        for k=1:modelsNumber
            output = models{k}.predict(xtets);
            accuracy{iter,k} = Evaluate(ytest,output); % EVAL = [accuracy sensitivity specificity precision recall f_measure gmean]
        end
        
        output = fitcsvmMODEL.predict(xtets);
        accuracy{iter,4} = Evaluate(ytest,output); % EVAL = [accuracy sensitivity specificity precision recall f_measure gmean]
        
        % Save the variable of interest of the current workspace
        if saveResult
            varFile = strcat(path,'\gridSearch.mat');
            save(varFile,'trainingStats','kernelEval','iterNumber','svNumber','accuracy','modelParam','violation');
        end
    end
end