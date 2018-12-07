% SVM using Sequential Minimal Optimization (SMO)
clear, clc;
load iris.dat;
iris(:,1:end-1)=zscore(iris(:,1:end-1));

%Select only element (1 and 2 attrubyte) of class 1 and 2
iris = iris(iris(:,end) <= 2,[1,2,end]);
iris( iris(:,end) == 2, end) = -1;

%Shuffle del dataset
s = RandStream('mt19937ar','Seed',0);
rand_pos = randperm(s, size(iris,1)); %array of random positions
irisShuffle = iris;
for i=1:size(iris,1)
    irisShuffle(i,:) = iris(rand_pos(i),:);
end

iris = irisShuffle;

%Divide dataset in training and testing data
p = .7;     % proportion of rows to select for training
N = size(iris,1);  % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;    
tf = tf(randperm(s,N));   % randomise order
dataTraining = iris(tf,:);
dataTesting = iris(~tf,:);


xTrain = dataTraining(:,1:end-1);
yTrain = dataTraining (:,end);


tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
eps =0.001;
C = inf;


smo = smo(xTrain, yTrain, C);
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