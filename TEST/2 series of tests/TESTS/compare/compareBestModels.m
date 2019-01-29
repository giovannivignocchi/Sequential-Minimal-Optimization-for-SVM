close all;
clear all;
clc;

load('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\TEST\on real dataset\TESTS RESULTS\GRID SERACH\RINGNORM\gridSearch.mat');
accFLC = cell2mat(accuracy(:,1));
accFLC = accFLC(:,1);
accJ4 = cell2mat(accuracy(:,2));
accJ4 = accJ4(:,1);
accJ6 = cell2mat(accuracy(:,3));
accJ6 = accJ6(:,1);

j = 0;
for i = [10 20 30 40 50]
    
    iterNumber1 = iterNumber(1+j:i,:);
    kernelEval1 = kernelEval(1+j:i,:);
    trainingStats1 = trainingStats(1+j:i,:);

    figure();
    subplot(2,4,1)
    plot(iterNumber1(:,1));
    hold on;
    plot(iterNumber1(:,2));
    hold on;
    plot(iterNumber1(:,3));
    title('Number of iteration')
    xlabel('sigma');
    xticks([1 5 10]);
    xticklabels({'2^-15', '2^-7', '2^3'});
    ylabel('Number of iterations log(10)');

    subplot(2,4,2);
    plot(trainingStats1(:,1));
    hold on;
    plot(trainingStats1(:,2));
    hold on;
    plot(trainingStats1(:,3));
    title('Training Time')
    xlabel('sigma');
    xticks([1 5 10]);
    xticklabels({'2^-15', '2^-7', '2^3'});
    ylabel('ms');

    subplot(2,4,3);
    plot(kernelEval1(:,1) ./ iterNumber1(:,1));
    hold on;
    plot(kernelEval1(:,2) ./ iterNumber1(:,2));
    hold on;
    plot(kernelEval1(:,3) ./ iterNumber1(:,3));
    title('Avg kernel eval/iter')
    xlabel('sigma');
    xticks([1 5 10]);
    xticklabels({'2^-15', '2^-7', '2^3'});


    subplot(2,4,4);
    plot(trainingStats1(:,1) ./ iterNumber1(:,1));
    hold on;
    plot(trainingStats1(:,2) ./ iterNumber1(:,2));
    hold on;
    plot(trainingStats1(:,3) ./ iterNumber1(:,3));
    title('Avg Training Time/iter')
    xlabel('sigma');
    xticks([1 5 10]);
    xticklabels({'2^-15', '2^-7', '2^3'});
    ylabel('ms');
    
    j = j + 10;
end
    