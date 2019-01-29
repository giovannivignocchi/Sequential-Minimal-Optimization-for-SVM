function plotStatistics(saveResult, path, C, sigma, modelParam, accFLC, iterFLC, svNumFLC, kernelEvalFLC, accJ4, iterJ4, svNumJ4, kernelEvalJ4, accJ6, iterJ6, svNumJ6, kernelEvalJ6)

xtickLab = strings(1,size(C,1));
for i=1:size(C,1)
    x = log2(C(i));
    x = strcat('2^',num2str(x));
    xtickLab(1,i) = x; 
end

% Still Need to fix the number of yLabel to be shown
ytickLab = strings(1,size(sigma,1));
for i=1:size(sigma,1)
    x = log2(sigma(i));
    x = strcat('2^',num2str(x));
    ytickLab(1,i) = x; 
end

iterFLC = flipud( reshape(iterFLC,size(sigma,1),size(C,1)) );
iterJ4 = flipud( reshape(iterJ4,size(sigma,1),size(C,1)) );
iterJ6 = flipud( reshape(iterJ6,size(sigma,1),size(C,1)) );

% Number of iterations
figure();
surf(iterFLC);
title('Number of iterations performed (maxiter = 10000)');
xlabel('C');
ylabel('sigma');
zlabel('iterations log(10)');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'iterFLC'), 'jpeg');
end

figure();
surf(iterJ4);
title('Number of iterations performed (maxiter = 10000)');
xlabel('C');
ylabel('sigma');
zlabel('iterations log(10)');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'iterJ4'), 'jpeg');
end

figure();
surf(iterJ6);
title('Number of iterations performed (maxiter = 10000)');
xlabel('C');
ylabel('sigma');
zlabel('iterations log(10)');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'iterJ6'), 'jpeg');
end

% Accuracy

% Get the best model found so far
[accMaxFLC,maxFLC] = maxk(accFLC,3);
[accMaxJ4,maxJ4] = maxk(accJ4,3);
[accMaxJ6,maxJ6] = maxk(accJ6,3);

bestModelFLC = zeros(3,2);
bestModelJ4 = zeros(3,2);
bestModelJ6 = zeros(3,2);

for i=1:3
    bestModelFLC(i,1) = modelParam(maxFLC(i),1);
    bestModelFLC(i,2) = modelParam(maxFLC(i),2);
end
for i=1:3
    bestModelJ4(i,1) = modelParam(maxJ4(i),1);
    bestModelJ4(i,2) = modelParam(maxJ4(i),2);
end
for i=1:3
    bestModelJ6(i,1) = modelParam(maxJ6(i),1);
    bestModelJ6(i,2) = modelParam(maxJ6(i),2);
end
 
accFLC = flipud( reshape(accFLC,size(sigma,1),size(C,1)) );
accJ4 = flipud( reshape(accJ4,size(sigma,1),size(C,1)) );
accJ6 = flipud( reshape(accJ6,size(sigma,1),size(C,1)) );

figure();
surf(accFLC);
title('Accuracy');
xlabel('C');
ylabel('sigma');
zlabel('accuracy');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^15'});
txt = compose('Best models found:\n1) C = %.2f, sigma = %f, accuracy = %.2f\n2) C = %.2f sigma = %f, accuracy = %.2f\n3) C = %.2f sigma = %f, accuracy = %.2f', bestModelFLC(1,1), bestModelFLC(1,2), accMaxFLC(1)*100,bestModelFLC(2,1), bestModelFLC(2,2), accMaxFLC(2)*100, bestModelFLC(3,1), bestModelFLC(3,2), accMaxFLC(3)*100);
annotation('textbox',[0, 0.95, 0, 0],'String',txt,'EdgeColor','none','FitBoxToText','on','FontSize',8)
if saveResult
    saveas(gcf, fullfile(path, 'accFLC'), 'jpeg');
end

figure();
surf(accJ4);
title('Accuracy');
xlabel('C');
ylabel('sigma');
zlabel('accuracy');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
txt = compose('Best models found:\n1) C = %.2f, sigma = %f, accuracy = %.2f\n2) C = %.2f sigma = %f, accuracy = %.2f\n3) C = %.2f sigma = %f, accuracy = %.2f', bestModelJ4(1,1), bestModelJ4(1,2), accMaxJ4(1)*100,bestModelJ4(2,1), bestModelJ4(2,2), accMaxJ4(2)*100, bestModelJ4(3,1), bestModelJ4(3,2), accMaxJ4(3)*100);
annotation('textbox',[0, 0.95, 0, 0],'String',txt,'EdgeColor','none','FitBoxToText','on','FontSize',8)
if saveResult
    saveas(gcf, fullfile(path, 'accJ4'), 'jpeg');
end

figure();
surf(accJ6);
title('Accuracy');
xlabel('C');
ylabel('sigma');
zlabel('accuracy');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
txt = compose('Best models found:\n1) C = %.2f, sigma = %f, accuracy = %.2f\n2) C = %.2f sigma = %f, accuracy = %.2f\n3) C = %.2f sigma = %f, accuracy = %.2f', bestModelJ6(1,1), bestModelJ6(1,2), accMaxJ6(1)*100, bestModelJ6(2,1), bestModelJ6(2,2), accMaxJ6(2)*100, bestModelJ6(3,1), bestModelJ6(3,2), accMaxJ6(3)*100);
annotation('textbox',[0, 0.95, 0, 0],'String',txt,'EdgeColor','none','FitBoxToText','on','FontSize',8)
if saveResult
    saveas(gcf, fullfile(path, 'accJ6'), 'jpeg');
end

% Number of support vector
svNumFLC = flipud( reshape(svNumFLC,size(sigma,1),size(C,1)) );
svNumJ4 = flipud( reshape(svNumJ4,size(sigma,1),size(C,1)) );
svNumJ6 = flipud( reshape(svNumJ6,size(sigma,1),size(C,1)) );

figure();
surf(svNumFLC);
title('Number of support vector genarted');
xlabel('C');
ylabel('sigma');
zlabel('num of support vector');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'svFLC'), 'jpeg');
end

figure();
surf(svNumJ4);
title('Number of support vector genarted');
xlabel('C');
ylabel('sigma');
zlabel('num of support vector');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'svJ4'), 'jpeg');
end

figure();
surf(svNumJ6);
title('Number of support vector genarted');
xlabel('C');
ylabel('sigma');
zlabel('num of support vector');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'svJ6'), 'jpeg');
end

% Number of kernel evaluation
kernelEvalFLC = flipud( reshape(kernelEvalFLC,size(sigma,1),size(C,1)) );
kernelEvalJ4 = flipud( reshape(kernelEvalJ4,size(sigma,1),size(C,1)) );
kernelEvalJ6 = flipud( reshape(kernelEvalJ6,size(sigma,1),size(C,1)) );

figure();
surf(kernelEvalFLC);
title('Number of kernel evaluation');
xlabel('C');
ylabel('sigma');
zlabel('num of ker evaluation log(10)');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'kerFLC'), 'jpeg');
end

figure();
surf(kernelEvalJ4);
title('Number of kernel evaluation');
xlabel('C');
ylabel('sigma');
zlabel('num of ker evaluation log(10)');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'kerJ4'), 'jpeg');
end

figure();
surf(kernelEvalJ6);
title('Number of kernel evaluation');
xlabel('C');
ylabel('sigma');
zlabel('num of ker evaluation log(10)');
xticklabels(xtickLab);
yticks([1 3 5 7 9 10]);
yticklabels({'2^3','2^-1','2^-5','2^-9','2^-13','2^-15'});
if saveResult
    saveas(gcf, fullfile(path, 'kerJ6'), 'jpeg');
end

end