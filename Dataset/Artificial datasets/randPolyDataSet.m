function [data, coeff, title] = randPolyDataSet(trainingSize, maxDegree, incorrectPercentage, delta)
%
% randPolyDataSet generate a random polynomial of maximum degre (maxDegree)
% and after that builds a two classes artificial dataset where the points
% belonging to different classes stand on different sides of the curve
% outlines by the random polynomial.
%
% If the parameter incorrectPercentage is not specified we obtain a
% perfectly separable dataset. If 0 < incorrectPercentage < 1 is specified
% a fraction of the points (equals to incorrectPercentage) is no more
% located on the correct side of the curve, doing so we obtain a dataset
% that is no more perfectly separable.
%
%
% INPUT ARGUMENTS
%
% trainingSize = (default 1000) is the size of the artificial dataset. If
%                trainingSize is odd then it is rounded down to first even number
% maxDegree = (default 5) is the maximum degree that the random polynomila
%              can atteins.
% incorrectPercentage = (default 0) is the percentage of points that is
%                        placed on the incorrect side of the curve
% delta = (default 300) is a displacement used to better distantiate points from the
%         polynomial curve, needed in case we want a separable dataset.


    if nargin < 1
        trainingSize = 1000;
    end
    
    if mod(trainingSize,2) ~= 0
        trainingSize = round(trainingSize/2) * 2;
    end

    if nargin < 2
        maxDegree = 5;
    end
    
    if nargin < 3
        incorrectPercentage = 0;
    end
    
    if incorrectPercentage < 0 || incorrectPercentage > 1
        disp("ERROR: incorrectPercentage must be a number between 0 and 1");
        return;
    end
    
    if nargin < 4
        delta = 300;
    end
    
    % Pick randomly the coefficients of the polynomial (coefficients are
    % constrained to lie between -5 and 5 to avoid high magnitude coefficients).
    coeff = randi([-5 5], 1, maxDegree);
    
    % Randomly picks the value of abscissa of the points, keeping divided
    % point belonging to different classes
    x1A = -10 + 20*rand(trainingSize/2,1);
    x1B = -10 + 20*rand(trainingSize/2,1);
    X1 = [x1A;x1B];

    % Evaluate the generated polynomial with the points in x1A and x1B 
    resA = zeros(trainingSize/2,1);
    resB = zeros(trainingSize/2,1);

    for i=1:size(x1A,1)
        resA(i) = polyval(coeff,x1A(i));
    end

    for i=1:size(x1B,1)
        resB(i) = polyval(coeff,x1B(i));
    end

    res =[resA ; resB];
    
    % Calculate the maximum values achive
    Max = max(res);
    Min = min(res);
    
    if abs(Max) > abs(Min)
        Min = -Max;
    else
        Max = -Min;
    end
    
    % Calculate the 2 displacement in which place the ordinate of the points
    GapA = Max .* ones(trainingSize/2,1) - resA; 
    GapB = resB - Min .* ones(trainingSize/2,1);

    % Randomply place the points
    x2A = delta + rand(trainingSize/2,1) .* GapA + resA;
    x2B = - delta - rand(trainingSize/2,1) .* GapB + resB;
    X2 = [x2A;x2B];
    
    
    % Calculate the number of points that need to be coorrectly separable and those that needn't.
    incorrectPoint = round(incorrectPercentage * trainingSize/2);
    correctPoint = trainingSize/2 - incorrectPoint;
    
    
    Y = [ones(correctPoint,1); -ones(incorrectPoint,1); ones(incorrectPoint,1); -ones(correctPoint,1)];
    data = [X1 X2 Y];

    %Shuffle del dataset
    rand_pos = randperm(size(data,1)); %array of random positions
    dataShuffle = data;

    for i=1:size(data,1)
        dataShuffle(i,:) = data(rand_pos(i),:);
    end
    
    data = dataShuffle;
    
    % Write the polynomial generated to be returned as a string
    title = "";
    for k=1:size(coeff,2)
        if k == 1
            formatSpec = " %d";
            A = coeff(1,end + 1 - k);
        elseif k == 2
            if coeff(1,end + 1 - k) < 0
                formatSpec = " %dX";
            else
                formatSpec = " +%dX";
            end
            A = coeff(1,end + 1 - k);
        elseif coeff(1,end + 1 - k) > 0
            formatSpec = " +%dX^%d";
            A = [coeff(1,end + 1 - k) k-1];
        elseif coeff(1,end + 1 - k) < 0
            formatSpec = " %d X^%d";
            A = [coeff(1,end + 1 - k) k-1];
        else
            continue;
        end
        str = compose(formatSpec,A);
        title = strcat(title,str);
    end
    
    
end