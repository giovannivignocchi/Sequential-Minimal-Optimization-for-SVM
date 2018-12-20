x = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\in_diabetes.txt');
y = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\out_diabetes.txt');
x = table2array(x);
y = table2array(y);

x = zscore(x);

y = [0;y];

%Shuffle del dataset
rand_pos = randperm(size(x,1)); %array of random positions
xTrainShuffle = x;
yTrainShuffle = y;


for i=1:size(x,1)
    xTrainShuffle(i,:) = x(rand_pos(i),:);
    yTrainShuffle(i,:) = y(rand_pos(i),:);
end
x = xTrainShuffle;
y = yTrainShuffle;

MdlDiabets = fitcsvm(x,y,'KernelFunction','rbf','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));



