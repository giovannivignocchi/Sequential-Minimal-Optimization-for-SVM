close all
clear all

load('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\TEST\on real dataset\TESTS RESULTS\GRID SERACH\MAGIC\gridSearch.mat');

C = [2^-3, 2, 2^9]';
sigma = [2^-15, 2^-9, 2^-3, 2, 2^3]';

accFLC = cell2mat(accuracy(:,1));
accFLC = accFLC(:,1);
accJ4 = cell2mat(accuracy(:,2));
accJ4 = accJ4(:,1);
accJ6 = cell2mat(accuracy(:,3));
accJ6 = accJ6(:,1);

saveResult = 0;
path = 'C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\TEST\on real dataset\TEST RESULTS\GRID SERACH\MAGIC\FIGURE';

plotStatistics(saveResult, path, C, sigma, modelParam, accFLC, log10(iterNumber(:,1)), svNumber(:,1), log10(kernelEval(:,1)), accJ4, log10(iterNumber(:,2)), svNumber(:,2), log10(kernelEval(:,2)), accJ6, log10(iterNumber(:,3)), svNumber(:,3), log10(kernelEval(:,3)));