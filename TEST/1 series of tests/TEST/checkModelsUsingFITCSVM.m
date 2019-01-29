function  SVMModel = checkModelsUsingFITCSVM(saveResult,path,xTrain,yTrain,C,tolerance,maxiter,kernel,x1Grid,x2Grid)

xGrid = [x1Grid(:),x2Grid(:)];

% Build the model using fitcsvm to control the results obtained
SVMModel = fitcsvm(xTrain,yTrain,'BoxConstraint',C,'KKTTolerance',tolerance,'IterationLimit',maxiter,'KernelFunction',kernel,'Solver','SMO');
svInd = SVMModel.IsSupportVector;
[~,score] = predict(SVMModel,xGrid);

% Plot the reults for the model generated using fitcsvm
figure('NumberTitle', 'off', 'Name', 'model obtained using fitcsvm');
subplot(2,2,1);
h(1:2) = gscatter(xTrain(:,1),xTrain(:,2),yTrain,'rb','.');
hold on
h(3) = plot(xTrain(svInd,1),xTrain(svInd,2),'ko');
contour(x1Grid,x2Grid,reshape(score(:,2),size(x1Grid)),[0 0],'k');
s=findobj('type','legend');
delete(s)
title("model generated");
axis equal
hold off

subplot(2,2,2);
contourf(x1Grid,x2Grid,reshape(score(:,2),size(x1Grid)),10);
hold on
contour(x1Grid,x2Grid,reshape(score(:,2),size(x1Grid)),[0 0],'k','LineWidth',2);
title('contour lines of the model');

if saveResult
    % Save current figure in the selected path
    saveas(gcf, fullfile(path, 'results using fitcsvm'), 'jpeg');
end

end

