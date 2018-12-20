x = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\in_magic.txt');
y = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\out_magic.txt');
x = x(:,1:end-1);
y = y(:,1:end-1);
x = table2array(x);
y = table2array(y);

x = zscore(x);

%Subsampling
x = [x(1:1000,:);x(end-1000:end,:)];
y = [y(1:1000,:);y(end-1000:end,:)];


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

MdlMagic = fitcsvm(x,y,'KernelFunction','rbf','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));



