%% TEST INITIALIZATION
clear all;
clc;
name = 'CORNER TEST';

% For reproducibility
seed = randi(100,1);
rng(seed);


saveResult = 1; % if set to 1 the results of the test will be stored
[path,fid] = initTest(saveResult, name, seed);


%% CREATE THE ARTIFICIAL DATASET
trainingSize = 250;
data = corners(trainingSize);

if saveResult
    fprintf(fid, 'DataSet parameter:\n');
    fprintf(fid, 'trainingSize: %d\n', trainingSize);
end

%Shuffle del dataset
s = RandStream('mt19937ar','Seed',0);
rand_pos = randperm(s, size(data,1)); %array of random positions
dataShuffle = data;
for i=1:size(data,1)
    dataShuffle(i,:) = data(rand_pos(i),:);
end

xTrain = dataShuffle(:,1:2);
yTrain = dataShuffle(:,3);

% Standardize the dataset
xTrain = zscore(xTrain);

% build the grid over which make preiction
dX1 = (max(xTrain(:,1)) - min(xTrain(:,1))) / 200;
dX2 = (max(xTrain(:,2)) - min(xTrain(:,2))) / 200;
[x1Grid,x2Grid] = meshgrid(min(xTrain(:,1)):dX1:max(xTrain(:,1)),min(xTrain(:,2)):dX2:max(xTrain(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

%% BUILDING MODELS

% Set parameter for the Models
C = inf;
tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
tau = 1e-12;
eps = 10e-5;
maxiter = 200;
kernel = 'gaussian';

if saveResult
    fprintf(fid, 'Model parameter:\n');
    fprintf(fid, 'C = %d\n', C);
    fprintf(fid, 'tolerance = %d\n', tolerance);
    fprintf(fid, 'tau = %d\n', tau);
    fprintf(fid, 'eps = %d\n', eps);
    fprintf(fid, 'maxiter = %d\n', maxiter);
    fprintf(fid, 'kernel = %s\n\n', kernel);
end

[models,figureTitle] = initModel(xTrain, yTrain, C, tolerance, eps, tau, maxiter, kernel);

% Keerthi = KeerthiSmo(xTrain, yTrain, C, tolerance, eps, maxiter);
% Keerthi.setKernel(kernel);
% 
% models = cell(1,1);
% models{1} = Keerthi;
% figureTitle = cell(1,1);
% figureTitle{1} = 'Keerthi';

output = zeros(size(xGrid,1),size(models,2));

trainingStats = cell(1, size(models,2));
predictionStats = cell(1, size(models,2));


for k=1:size(models,2)
    
    tic
    models{k}.train();
    trainingStats{k} = toc;
    
    tic
    output(:,k) = models{k}.predict(xGrid);
    predictionStats{k} = toc;
      
end

% Check the validity of the results obtained using fitcsvm
fitcsvmMODEL = checkModelsUsingFITCSVM(saveResult, path, xTrain,yTrain,C,tolerance,maxiter,kernel,x1Grid,x2Grid);

    
%% TEST RESULTS

%Write Test statistics
if saveResult
    for k=1:size(models,2)
        fprintf(fid, compose("------------------------------------------ %s ------------------------------------------\n", figureTitle{k}));

        fprintf(fid, 'Training time %f sec\n', trainingStats{k});

        fprintf(fid, 'Prediction time %f sec\n', predictionStats{k});
        
        fprintf(fid, 'Number of iteration %d\n',models{k}.iter);
        
        fprintf(fid, 'Average iteration time %f sec\n', trainingStats{k} / models{k}.iter);
        
        fprintf(fid, 'Number of support vector generated: %d\n', sum(models{k}.isSupportVector));
        
        fprintf(fid, 'Number of SV shared with fitcsvm model: %d\n\n', sum( and(models{k}.isSupportVector, SVMModel.IsSupportVector) ));
        
    end
    fclose(fid);
end


% Plot the reults for the generated models
plotResults(saveResult, path, name, models, figureTitle, output, x1Grid, x2Grid)

% Save the current workspace
if saveResult
    varFile = strcat(path,'\var.m');
    save(varFile);
end