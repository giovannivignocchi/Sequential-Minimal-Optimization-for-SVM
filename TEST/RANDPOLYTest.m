%% Build the dataset
clear all;

%Path in which the figure resulting from the test will be saved
path = 'C:\Users\giova\Desktop';

rng(24); % For reproducibility

maxDegree = 5; %degree of the polynomial
trainingSize = 200;

[data, coeff, polyTitle] = randPolyDataSet(trainingSize, maxDegree, 0);
xTrain = data(:,1:2);
yTrain = data(:,3);

%Standardize the dataset
xTrain = zscore(xTrain);

% Build the grid in which we are going to predict values
dX1 = (max(xTrain(:,1)) - min(xTrain(:,1))) / 500;
dX2 = (max(xTrain(:,2)) - min(xTrain(:,2))) / 500;
[x1Grid,x2Grid] = meshgrid(min(xTrain(:,1)):dX1:max(xTrain(:,1)),min(xTrain(:,2)):dX2:max(xTrain(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

%% Build the models

% Set parameter for the Models
tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
C = inf;

modelsNumber = 3;
models = cell(1, modelsNumber);
figureTitle = cell(1, modelsNumber);

smo = smo(xTrain, yTrain, C);
smo.setKernel('gaussian');
errorSmo = smoErrorCache(xTrain, yTrain, C);
errorSmo.setKernel('gaussian');
FCLsmo = FCLsmo(xTrain, yTrain, C);
FCLsmo.setKernel('gaussian');


models{1} = smo;
models{2} = errorSmo;
models{3} = FCLsmo;

figureTitle{1} = "PLATT version";
figureTitle{2} = "PLATT version with Error cache";
figureTitle{3} = "Fan Chen and Lin version";

output = zeros(size(xGrid,1),modelsNumber);

for k=1:size(models,2)
    
    tic
    models{k}.train();
    trainingStats(k) = toc;
    
    tic
    output(:,k) = models{k}.predict(xGrid);
    predictionStats(k) = toc;
    
    numberOfSV(k) = sum(models{k}.isSupportVector);
    
end

% %Statistichs about training time an prediction time
% disp("---------------------------------------------------------------")
% formatSpec = "Training time %f sec";
% str = compose(formatSpec,traningTime);
% disp(str);
% disp("---------------------------------------------------------------")
% formatSpec = "Prediction time %f sec";
% str = compose(formatSpec,predictionTime);
% disp(str);
% disp("---------------------------------------------------------------")

%% Plot the reult

for k=1:size(models,2)
    
    formatSpec = 'Random polynomial approximation TEST for %s';
    strTitle = compose(formatSpec,figureTitle{k});
    figure('NumberTitle', 'off', 'Name', strTitle{1});

    % Plot the data and the decision boundary
    subplot(2,2,1)
    h(1:2) = gscatter(xTrain(:,1),xTrain(:,2),yTrain,'rb','.');
    hold on
    h(3) = plot(xTrain(models{k}.isSupportVector,1),xTrain(models{k}.isSupportVector,2),'ko');
    contour(x1Grid,x2Grid,reshape(output(:,k),size(x1Grid)),[0 0],'k');
    s=findobj('type','legend');
    delete(s)
    title(polyTitle);
    axis equal
    hold off

    %Plot the heat map of the countour lines characterizing the function
    subplot(2,2,2)
    contourf(x1Grid,x2Grid,reshape(output(:,k),size(x1Grid)),10);
    hold on
    contour(x1Grid,x2Grid,reshape(output(:,k),size(x1Grid)),[0 0],'k','LineWidth',2);
    title('contour lines of the model');

    % Plot the behaviour of alphas during the iterations of the algorithm
    subplot(2,2,[3 4])
    supportVectorHistory = models{k}.alphaHistory(models{k}.isSupportVector,1:models{k}.iter);
    plot(supportVectorHistory');
    formatSpec = 'behaviour of LMs during algorithm iterations (maxIter = %d)';
    str = compose(formatSpec,models{k}.maxiter);
    title(str);
    
    % Save current figure in the selected path
    saveas(gcf, fullfile(path, figureTitle{k}), 'jpeg');
end