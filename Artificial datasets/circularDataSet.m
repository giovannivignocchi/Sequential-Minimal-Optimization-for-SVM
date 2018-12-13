function [xTrain,yTrain] = circularDataSet(trainingSize, incorrectPercentage, r1, r2)

    if nargin < 1
        trainingSize = 1000;
    end
    
    if mod(trainingSize,2) ~= 0
        trainingSize = round(trainingSize/2) * 2;
    end
    
    if nargin < 2
        incorrectPercentage = 0;
    end
    
    if incorrectPercentage < 0 || incorrectPercentage > 1
        disp("ERROR: incorrectPercentage must be a number between 0 and 1");
        return;
    end
    
    
    if nargin < 3
        r1 = 1;
    end
    
    if nargin < 4
        r2 = 3;
    end

    r = sqrt(r1*rand(trainingSize/2,1)); % Radius
    t = 2*pi*rand(trainingSize/2,1);  % Angle
    data1 = [r.*cos(t), r.*sin(t)]; % Points 1 class
    r2 = sqrt(r2*rand(trainingSize/2,1)+r1); % Radius
    t2 = 2*pi*rand(trainingSize/2,1);      % Angle
    data2 = [r2.*cos(t2), r2.*sin(t2)]; % Points 2 class
    xTrain = [data1;data2];

    % Calculate the number of points that need to be coorrectly separable and those that needn't.
    incorrectPoint = round(incorrectPercentage * trainingSize/2);
    correctPoint = trainingSize/2 - incorrectPoint;
    
    
    yTrain = [ones(correctPoint,1); -ones(incorrectPoint,1); ones(incorrectPoint,1); -ones(correctPoint,1)];

    % Standardize the dataset
    xTrain = zscore(xTrain);

    % Shuffle del dataset
    s = RandStream('mt19937ar','Seed',0);
    rand_pos = randperm(s, size(xTrain,1)); %array of random positions
    xTrainShuffle = xTrain;
    yTrainShuffle = yTrain;
    for i=1:size(xTrain,1)
        yTrainShuffle(i,1) = yTrain(rand_pos(i));
        xTrainShuffle(i,:) = xTrain(rand_pos(i),:);
    end

    xTrain = xTrainShuffle;
    yTrain = yTrainShuffle;

end

