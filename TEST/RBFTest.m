%% Build the model
clear all;
clc;

%rng(10); % For reproducibility

r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points
r2 = sqrt(3*rand(100,1)+1); % Radius
t2 = 2*pi*rand(100,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points
xTrain = [data1;data2];
yTrain = ones(200,1);
yTrain(1:100) = -1;

%Standardize the dataset
xTrain = zscore(xTrain);

%Shuffle del dataset
s = RandStream('mt19937ar','Seed',0);
rand_pos = randperm(s, size(xTrain,1)); %array of random positions
xTrainShuffle = xTrain;
yTrainShuffle = yTrain;
for i=1:size(xTrain,1)
    yTrainShuffle(i,1) = yTrain(rand_pos(i));
    xTrainShuffle(i,:) = xTrain(rand_pos(i),:);
end

xTrain = xTrainShuffle;
yTrain = yTrainShuffle;

% build the grid over which make preiction
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(xTrain(:,1)):d:max(xTrain(:,1)),min(xTrain(:,2)):d:max(xTrain(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
C = inf;

models = cell(1, 2);
figureTitle = cell(1, 2);

smo = smo(xTrain, yTrain, C);
smo.setKernel('gaussian');
FCLsmo = FCLsmo(xTrain, yTrain, C);
FCLsmo.setKernel('gaussian');

models{1} = smo;
models{2} = FCLsmo;

figureTitle{1} = "PLATT version";
figureTitle{2} = "Fan Chen and Lin version";

output = zeros(size(xGrid,1),2);

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
% formatSpec = "Training time %d";
% str = compose(formatSpec,traningTime);
% disp(str);
% disp("---------------------------------------------------------------")
% formatSpec = "Prediction time %d";
% str = compose(formatSpec,predictionTime);
% disp(str);
% disp("---------------------------------------------------------------")

%% Plot the reult
for k=1:size(models,2)
    
    formatSpec = 'RBF TEST for %s';
    strTitle = compose(formatSpec,figureTitle{k});
    figure('NumberTitle', 'off', 'Name', strTitle{1});

    % Plot the data and the decision boundary
    subplot(2,2,1)
    h(1:2) = gscatter(xTrain(:,1),xTrain(:,2),yTrain,'rb','.');
    hold on
    h(3) = plot(xTrain(models{k}.isSupportVector,1),xTrain(models{k}.isSupportVector,2),'ko');
    contour(x1Grid,x2Grid,reshape(output(:,k),size(x1Grid)),[0 0],'k');
    %legend(h,{'-1','+1','Support Vectors'});
    s=findobj('type','legend');
    delete(s)
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
end