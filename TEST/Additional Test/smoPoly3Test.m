clear all;
clc;

%rng(10); % For reproducibility

%coefficients of the function
a = 0;
while a == 0
    a = randi([-3, 3],1);
end
b = randi([-3, 3],1);
c = randi([-3, 3],1);
d = randi([-2, 2],1);

trainingSize = 100;

x1A = -10 + 20*rand(trainingSize/2,1);
x1B= -10 + 20*rand(trainingSize/2,1);
X1 = [x1A;x1B];

Max = max((a .* X1.^3 + b .* X1.^2 + c .* X1 + d));
Min = min((a .* X1.^3 + b .* X1.^2 + c .* X1 + d));

GapA = Max .* ones(trainingSize/2,1) - (a .* x1A.^3 + b .* x1A.^2 + c .* x1A + d); 
GapB = (a .* x1B.^3 + b .* x1B.^2 + c .* x1B + d) - Min .* ones(trainingSize/2,1);

% Distance use to strongly keep distantiate the two classes.
delta = 100;

x2A = delta + rand(trainingSize/2,1) .* GapA + (a .* x1A.^3 + b .* x1A.^2 + c .* x1A + d);
x2B = - delta - rand(trainingSize/2,1) .* GapB + (a .* x1B.^3 + b .* x1B.^2 + c .* x1B + d);
X2 = [x2A;x2B];

xTrain = [X1 X2];
yTrain = [ones(trainingSize/2,1);-ones(trainingSize/2,1)];

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

tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
C = inf;

smo = smo(xTrain, yTrain, C);
smo.setKernel('polynomial',3);
smo.train();

% Predict scores over the grid
dX1 = (max(xTrain(:,1)) - min(xTrain(:,1))) / 500;
dX2 = (max(xTrain(:,2)) - min(xTrain(:,2))) / 500;
[x1Grid,x2Grid] = meshgrid(min(xTrain(:,1)):dX1:max(xTrain(:,1)),min(xTrain(:,2)):dX2:max(xTrain(:,2)));
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
hold on
contour(x1Grid,x2Grid,reshape(output,size(x1Grid)),[0 0],'k','LineWidth',2);

% Plot the behaviour of alphas during the iterations of the algorithm
figure;
supportVectorHistory = smo.alphaHistory(smo.isSupportVector,1:smo.iter);
plot(supportVectorHistory');