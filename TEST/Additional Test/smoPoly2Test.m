clear all;
clc;

N = 200;

data = crescentfullmoon(N);

%Shuffle del dataset
s = RandStream('mt19937ar','Seed',0);
rand_pos = randperm(s, size(data,1)); %array of random positions
xTrainShuffle = data;
for i=1:size(data,1)
    xTrainShuffle(i,:) = data(rand_pos(i),:);
end

data = xTrainShuffle;
xTrain = data(:,1:2);
yTrain = data(:,3);

tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
C = inf;

smo = smo(xTrain, yTrain, C);
smo.setKernel('gaussian');
smo.train();

% Predict scores over the grid
d = 0.2;
[x1Grid,x2Grid] = meshgrid(min(xTrain(:,1)):d:max(xTrain(:,1)),min(xTrain(:,2)):d:max(xTrain(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
output = smo.predict(xGrid);

% Plot the data and the decision boundary
figure;
h(1:2) = gscatter(xTrain(:,1),xTrain(:,2),yTrain,'rb','.');
hold on
h(3) = plot(xTrain(smo.isSupportVector,1),xTrain(smo.isSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(output,size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off

%Plot the heat map of the countour lines characterizing the function
figure;
contourf(x1Grid,x2Grid,reshape(output,size(x1Grid)),10);

% Plot the behaviour of alphas during the iterations of the algorithm
figure;
supportVectorHistory = smo.alphaHistory(smo.isSupportVector,1:smo.iter);
plot(supportVectorHistory');