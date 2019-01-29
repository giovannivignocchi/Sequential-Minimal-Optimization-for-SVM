function [path,fid] = initTest(saveResult, name, testDirectory, seed)

% Add path of external function used during the test
addpath('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Algorithm\Platt');
addpath('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Algorithm\Fan Chen and Lin');
addpath('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Algorithm\Keerthi');
addpath('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Algorithm\Joachims');
addpath('C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Artificial datasets');

%Path in which results of the TEST will be saved
CurrentDate = date;
testName = compose('%s %s rng(%d)', name, CurrentDate, seed);
path = strcat(testDirectory,testName);

% Create the corresponding TEST directory
if saveResult
    mkdir(path);
    
    % Create txt file to save test statistics in the created folder
    statFilePath = strcat(path,'\stat.txt');
    fid = fopen( statFilePath, 'wt' );
    fprintf(fid, '%s\n\n', testName{1});
else
    fid = 0;
end

end

