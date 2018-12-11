%% TEST INITIALIZATION
clear all;
name = 'LINEAR TEST';

% For reproducibility
seed = 12;
rng(seed);


saveResult = 0; % if set to 1 the results of the test will be stored
[path,fid] = initTest(saveResult, name, seed);

%% CREATE THE ARTIFICIAL DATASET
trainingSize = 200;
maxDegree = 2;
incorrectPercentage = 0;
delta = 5;
[data, coeff, title] = randPolyDataSet(trainingSize, maxDegree, incorrectPercentage, delta);
xTrain = data(:,1:2);
yTrain = data(:,3);

if saveResult
    fprintf(fid, 'DataSet parameter:\n');
    fprintf(fid, 'polynomial that guide dataset genartion: = %s\n', polyTitle);
    fprintf(fid, 'trainingSize = %d\n', trainingSize);
    fprintf(fid, 'maxDegree = %d\n', maxDegree);
    fprintf(fid, 'incorrectPercentage = %d\n', incorrectPercentage);
    fprintf(fid, 'delta = %d\n\n', delta);
end

%Standardize the dataset
xTrain = zscore(xTrain);

% Predict scores over the grid
dX1 = (max(xTrain(:,1)) - min(xTrain(:,1))) / 500;
dX2 = (max(xTrain(:,2)) - min(xTrain(:,2))) / 500;
[x1Grid,x2Grid] = meshgrid(min(xTrain(:,1)):dX1:max(xTrain(:,1)),min(xTrain(:,2)):dX2:max(xTrain(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];


%% BUILDING MODELS

% Set parameter for the Models
C = inf;
tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
tau = 1e-12;
eps = 10e-5;
maxiter = 200;
kernel = 'linear';

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
output = zeros(size(xGrid,1),size(models,2));

trainingStats = cell(1, size(models,2));
predictionStats = cell(1, size(models,2));
numberOfSV = cell(1, size(models,2));

for k=1:size(models,2)
    
    tic
    models{k}.train();
    trainingStats{k} = toc;
    
    tic
    output(:,k) = models{k}.predict(xGrid);
    predictionStats{k} = toc;
    
    numberOfSV{k} = sum(models{k}.isSupportVector);   
    
end

% Save the current workspace
if saveResult
    varFile = strcat(path,'\var.m');
    save varFile;
end


%% TEST RESULTS

%Write Test statistics
if saveResult
    for k=1:size(models,2)
        fprintf(fid, compose("------------------------------------------ %s ------------------------------------------\n", figureTitle{k}));

        fprintf(fid, 'Training time %f sec\n', trainingStats{k});

        fprintf(fid, 'Prediction time %f sec\n', predictionStats{k});

        fprintf(fid, 'Number of support vector generated: %d\n', numberOfSV{k});
    end
    fclose(fid);
end


% Plot the reults for the generated models
plotResults(saveResult, path, name, models, figureTitle, output, x1Grid, x2Grid)