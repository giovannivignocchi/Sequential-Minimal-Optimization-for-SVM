x = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\in_ringnorm.txt');
y = readtable('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\out_ringnorm.txt');
x = x(:,1:end-1);
y = y(:,1:end-1);
x = table2array(x);
y = table2array(y);

x = zscore(x);

%Subsampling
x = x(1:1000,:);
y = y(1:1000,:);

MdlRingnorm = fitcsvm(x,y,'KernelFunction','rbf','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));



