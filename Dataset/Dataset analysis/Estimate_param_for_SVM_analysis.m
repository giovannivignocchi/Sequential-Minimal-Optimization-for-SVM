path = 'C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Dataset analysis';
set(0,'DefaultFigureVisible','off');

numberOfDataset = 4;
saveResult = 1;
iterationNumber = 100;
subsampleSize = 2000;

C = zeros(iterationNumber,numberOfDataset);
sigma = zeros(iterationNumber,numberOfDataset);

for i=1:iterationNumber
    seed = i;
    
    mdl = diabetes(subsampleSize,seed);
    C(i,1) = mdl.ModelParameters.BoxConstraint;
    sigma(i,1) = mdl.ModelParameters.KernelScale;
    
    mdl = in_codrna(subsampleSize,seed);
    C(i,2) = mdl.ModelParameters.BoxConstraint;
    sigma(i,2) = mdl.ModelParameters.KernelScale;
    
    mdl = magic(subsampleSize,seed);
    C(i,3) = mdl.ModelParameters.BoxConstraint;
    sigma(i,3) = mdl.ModelParameters.KernelScale;
    
    mdl = ringnorm(subsampleSize,seed);
    C(i,4) = mdl.ModelParameters.BoxConstraint;
    sigma(i,4) = mdl.ModelParameters.KernelScale;
    
end  

Cestimated = mean(C,1);
Sigmaestimated = mean(sigma,1);
CVar = var(C,0,1);
SigmaVar = var(sigma,0,1);


%Save the current workspace
if saveResult
    varFile = strcat(path,'\var.mat');
    save(varFile,'C','sigma','Cestimated','Sigmaestimated');
end

set(0,'DefaultFigureVisible','on');