function  plotResults(saveResult, path, name, models, figureTitle, output, x1Grid, x2Grid)


% Plot the reults for the generated models
for k=1:size(models,2)
    
    
    formatSpec = '%s for %s';
    strTitle = compose(formatSpec, name, figureTitle{k});
    figure('NumberTitle', 'off', 'Name', strTitle{1});

    % Plot the data and the decision boundary
    subplot(2,2,1)
    h(1:2) = gscatter(models{k}.x(:,1),models{k}.x(:,2),models{k}.y,'rb','.');
    hold on
    h(3) = plot(models{k}.x(models{k}.isSupportVector,1),models{k}.x(models{k}.isSupportVector,2),'ko');
    contour(x1Grid,x2Grid,reshape(output(:,k),size(x1Grid)),[0 0],'k');
    s=findobj('type','legend');
    delete(s)
    title("model generated");
    axis equal
    hold off

    %Plot the heat map of the countour lines characterizing the function
    subplot(2,2,2)
    contourf(x1Grid,x2Grid,reshape(output(:,k),size(x1Grid)),10);
    hold on
    contour(x1Grid,x2Grid,reshape(output(:,k),size(x1Grid)),[0 0],'k','LineWidth',2);
    title('contour lines of the model');

    % Plot the behaviour of alphas during the iterations of the algorithm
    subplot(2,2,[3 4])
    supportVectorHistory = models{k}.alphaHistory(models{k}.isSupportVector,1:models{k}.iter);
    plot(supportVectorHistory');
    formatSpec = 'behaviour of LMs during algorithm iterations (maxIter = %d)';
    str = compose(formatSpec,models{k}.maxiter);
    title(str);
    
    if saveResult
        % Save current figure in the selected path
        saveas(gcf, fullfile(path, figureTitle{k}), 'jpeg');
    end
    
end




end

