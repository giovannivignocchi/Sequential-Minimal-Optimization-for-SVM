classdef smoErrorCache < handle
    %SMO class 
    %
    %   Implemets the Sequential Minimal Optimization algorithm in the
    %   version suggested by Platt in the paper "Fast training of Support
    %   Vector machines using sequential minimal optimization"
    %   An online free version of the paper is available at:
    %   www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/
    %
    %   This first version of the algorithm does exploit the Error cache
    %   optimizatin proposed by Platt to speed up the computation
    %
    %
    %   PROPERTIES:
    %
    %   x = train set
    %   y = class label associated with the train set
    %   N = size of the train set
    %   alpha = vector of solution
    %   b = bias
    %   C = determines the tradeoff between increasing the margin-size and 
    %       ensuring that the x lie on the correct side of the margin.
    %   errorCache = cache containing prediction error for every training
    %                sample
    %   tolerance = (default 1e-3) tolerance in the strenght the KKT 
    %               conditions are fullfil.
    %   eps = (default 10e-5) treshold that has to be reached for an update
    %         to be valid. [alphaNew - alphaOld > eps]
    %   iter = number of iterations the training last.
    %   maxiter = (default 200) maximum number of iteration that the training
    %              algorithm is allow to run.
    %   kernelType = (default 'linear') indicate which kind of kernel is used
    %                during the training procedure. It is possible to set
    %                anoter type of kernal via setKernel method.
    %   degree = (default 2) degree of the polynomila kernel, is it possible
    %            to modify it using setKernel method.
    %   sigma = (default 1) sigma parameter used in the gaussian kernel, is
    %           it possible to modify it using setKernel method.
    %   isSupportVector = boolean vector that indicates which of the alpha
    %                     is effectively a support vector.
    %   kernelEvaluation = varable that records the number of kernel
    %                      evaluation carried out during the iteration of
    %                      the algorithm.
    
    properties
        x;
        y;
        N;
        alpha;
        bias;
        C;
        errorCache;
        
        tolerance = 1e-5;
        eps = 10e-5;
        iter = 0;
        maxiter = 200;
        
        kernelType = 'linear';
        degree = 2;
        sigma = 1;
        
        isSupportVector;
        kernelEvaluation = 0;
    end
    
    methods
        
        function obj = smoErrorCache(data, classLabels, C, tolerance, eps, maxiter)
            % SMO Constructor
            
            % Checking optional parameter
            if nargin < 3
                disp("ERROR: Not enough input arguments, not possible to instatitate smo");
                return;
            end
            
            if nargin > 3
                obj.tolerance = tolerance;
            end
            
            if nargin > 4
                obj.eps = eps;
            end
            
            if nargin > 5
                obj.maxiter = maxiter;
            end
            
            obj.x = data;
            obj.y = classLabels;
            obj.C = C;
            obj.N = size(classLabels,1);
            
            % Initialize all Lagrange multipliers (LMs) to 0
            obj.alpha = zeros(obj.N,1);
            
            % Initialize error cache for each LMs to 0
            obj.errorCache = - obj.y;
            
            % initialize threshold to zero
            obj.bias = 0;
            
            obj.isSupportVector = zeros(obj.N,1);
        end
        
        function Ei = calcEi(smo,i)
            res = zeros(smo.N,1);
            for k=1:smo.N
                res(k) = smo.kernel(smo.x(i,:),smo.x(k,:));
            end
            u = sum ( smo.alpha .* smo.y .* res) - smo.bias;
            Ei = u - smo.y(i);
        end
       
        function [L,H] = calculateBoundaries(smo,i1,i2,alphaOld1,alphaOld2)
            if (smo.y(i1) ~= smo.y(i2) )
                L = max(0, alphaOld2 - alphaOld1);
                H = min(smo.C, smo.C + alphaOld2 - alphaOld1);
            else
                L = max(0, alphaOld2 + alphaOld1 - smo.C);
                H = min(smo.C, alphaOld2 + alphaOld1);
            end
        end
        
        function ker = kernel(smo,x1,x2)
            if  strcmp(smo.kernelType,'gaussian')
                ker = exp(-norm(x2-x1).^2 * smo.sigma); %gaussian Kernel
            elseif strcmp(smo.kernelType,'polynomial')
                ker = (1 + x1 * x2').^smo.degree; %polynomial Kernel
            else
                ker = x1*x2'; %linear Kernel
            end
            
            smo.kernelEvaluation = smo.kernelEvaluation + 1;
        end
        
        function setKernel(smo,type,varargin)
            if size(varargin,2) > 1
                disp("ERROR: Too many input argument in function setKernel");
                return;
            end
            if strcmp(type,'gaussian')
                smo.kernelType = 'gaussian';
                
                if size(varargin,2) == 1 
                    smo.sigma = varargin{1};
                end
            elseif strcmp(type,'polynomial')
                smo.kernelType = 'polynomial';
                if size(varargin,2) == 1
                    smo.degree = varargin{1};
                end
            end
        end
        
        function update = takeStep(smo,i1,E1,i2)
            
            update = 0;
            
            if (i1 == i2)
                return;
            end
            
            s = smo.y(i1) * smo.y(i2);
                        
            if (smo.alpha(i2) > 0 && smo.alpha(i2) < smo.C)
                E2 = smo.errorCache(i2);
            else
                E2 = smo.calcEi(i2);
                smo.errorCache(i2) = E2;
            end
           
            alphaOld1 = smo.alpha(i1);
            alphaOld2 = smo.alpha(i2);
            
            [Low, High] = smo.calculateBoundaries(i1,i2,alphaOld1,alphaOld2);
            
            if (Low == High)
                return;
            end
            
            k11 = smo.kernel(smo.x(i1,:),smo.x(i1,:));
            k12 = smo.kernel(smo.x(i1,:),smo.x(i2,:));
            k22 = smo.kernel(smo.x(i2,:),smo.x(i2,:));
            eta = 2 * k12 - k11 - k22;
            
            if (eta < 0)
                
                alphaNew2 = alphaOld2 + smo.y(i2) * (E2 - E1) / eta;

                if(alphaNew2 > High)
                    alphaNew2 = High;
                end
                if(alphaNew2 < Low)
                    alphaNew2 = Low;
                end
                
            % If eta >= 0, the obj. function should be avaluated at eachend of the line segment.
            % SMO moves the LM to the end point with the highest value of the obj.
            % function, if the value is the same at both ends (within a small eps for round error)
            % and the kernel obey the Mercers's conditions, the joint maximization cannot make progress
            else
                c1 = eta/2;
                c2 = smo.y(i2) * (E1-E2) - eta * alphaOld2;
                Lobj = c1 * Low * Low + c2 * Low;
                Hobj = c1 * High * High + c2 * High;
                
                if (Lobj > Hobj)
                    alphaNew2 = Low;
                elseif (Lobj < Hobj)
                    alphaNew2 = High;
                else
                    alphaNew2 = alphaOld2;
                end
            end
            
            if abs(alphaNew2-alphaOld2) < smo.eps
                return;
            end
            
            %After this point the old LMs will be updated to a1 and a2
            update = 1;
            
            %Since a1 + s * a2 = alph1 + s * alph2
            alphaNew1 = alphaOld1 + s * (alphaOld2 - alphaNew2);
            
            if (alphaNew1 < 0)
                alphaNew2 = alphaNew2 + s * alphaNew1;
                alphaNew1 = 0;  
            elseif (alphaNew1 > smo.C)
                t = alphaNew1-smo.C;
                alphaNew2 = alphaNew2 + s * t;
                alphaNew1 = smo.C;
            end
            
            if (alphaNew1 > 0 && alphaNew1 < smo.C)
                bNew = smo.bias + E1 + smo.y(i1) * (alphaNew1 - alphaOld1) * k11 + smo.y(i2) * (alphaNew2 - alphaOld2) * k12;
            elseif (alphaNew2 > 0 && alphaNew2 < smo.C)
                bNew = smo.bias + E2 + smo.y(i2) * (alphaNew2 - alphaOld2) * k22 + smo.y(i1) * (alphaNew1 - alphaOld1) * k12;
            else
                %If both a1 and a2 take values 0 or C, the original SMO algorithm computes
                %two values of the new b for a1 and a2 and takes the average,
                b1 = smo.bias + E1 + smo.y(i1) * (alphaNew1 - alphaOld1) * k11 + smo.y(i2) * (alphaNew2 - alphaOld2) * k12;
                b2 = smo.bias + E2 + smo.y(i2) * (alphaNew2 - alphaOld2) * k22 + smo.y(i1) * (alphaNew1 - alphaOld1) * k12;
                bNew = (b1 + b2) / 2;
            end
            
            %Update the treshold b
            bOld = smo.bias;
            smo.bias = bNew;
            
            %Update the vector of LMs
            smo.alpha(i1) = alphaNew1;
            smo.alpha(i2) = alphaNew2;  
            
            %Update errorCache
            t1 = smo.y(i1)*(alphaNew1 - alphaOld1);
            t2 = smo.y(i2)*(alphaNew2 - alphaOld2);
            
            deltaB = bNew - bOld;
            
            for k=1:smo.N
                if (smo.alpha(k) > 0 && smo.alpha(k) < smo.C)
                    k1 = smo.kernel(smo.x(i1,:),smo.x(k,:));
                    k2 = smo.kernel(smo.x(i2,:),smo.x(k,:));
                    smo.errorCache(k) = smo.errorCache(k) + t1 * k1 + t2 * k2 - deltaB;
                end
            end

        end
        
        function numChanged = examineExample(smo,i1)
            
            numChanged = 0;
            
            if (smo.alpha(i1) > 0 && smo.alpha(i1) < smo.C)
                E1 = smo.errorCache(i1);
            else
                E1 = smo.calcEi(i1);
                smo.errorCache(i1) = E1;
            end
            
            r1 = smo.y(i1) * E1;
            
            % Check if i1 violates KKT condition
            if ((r1 < -smo.tolerance && smo.alpha(i1) < smo.C) || (r1 > smo.tolerance && smo.alpha(i1) > 0))
                
                i2 = -1; % index of the 2nd element to be optimized.
                
                % Apply a hyerarchy of heuristic to choose the second example  
            
                % 1) From non-bound examples, so that E1-E2 is maximized.
                max = 0;
                for k = 1 : smo.N
                    if (smo.alpha(k) > 0 && smo.alpha(k) < smo.C)
                        E2 = smo.errorCache(k);
                        diff = abs(E1 - E2);
                        if (diff > max)
                            max = diff;
                            i2 = k;
                        end
                    end
                end
                
                if(i2 >= 0)
                    if(takeStep(smo,i1,E1,i2))
                        numChanged = 1;
                        return;
                    end
                end
                
                % 2) If we cannot make progress with the best non-bound example, then try any non-bound examples
                %    (start iterating at random position in order not to bias smo towards example at the beginnig of the dataset)
                startIndex = randi([1, smo.N],1);
                for k=startIndex:smo.N
                    i2 = k;
                    if (smo.alpha(i2) > 0 && smo.alpha(i2) < smo.C)
                        if(takeStep(smo,i1,E1,i2))
                            numChanged = 1;
                            return;
                        end
                    end
                end
                % Repeat the same iteration for the index before start index, if takeStep have not succeeded yet 
                for k=1:startIndex
                    i2 = k;
                    if (smo.alpha(i2) > 0 && smo.alpha(i2) < smo.C)
                        if(takeStep(smo,i1,E1,i2))
                            numChanged = 1;
                            return;
                        end
                    end
                end
                
                % 3) If we cannot make progress with the non-bound examples, then try any example.
                %    (start iterating at random position in order not to bias smo towards example at the beginnig of the dataset)
                startIndex = randi([1, smo.N],1);
                for k=startIndex:smo.N
                    i2 = k;
                    if(takeStep(smo,i1,E1,i2))
                        numChanged = 1;
                        return;
                        
                    end
                end
                % Repeat the same iteration for the index before start index, if takeStep have not succeeded yet
                for k=1:startIndex
                    i2 = k;
                    if(takeStep(smo,i1,E1,i2))
                        numChanged = 1;
                        return;
                        
                    end
                end
            end
        end
        
        function train(smo)
            numChanged = 0;
            examineAll = 1;
            while (numChanged > 0 && smo.iter < smo.maxiter) || (smo.iter == 0)
                numChanged = 0;
                smo.iter = smo.iter + 1;
                if(examineAll)
                    %Loop over all training example
                    for i1=1:smo.N
                        numChanged = numChanged + examineExample(smo,i1);
                    end
                else
                    %Loop over all non-bounding example ( 0 < alpha < C)
                    for i1=1:smo.N
                        if (smo.alpha(i1) ~= 0 && smo.alpha(i1) ~= smo.C)
                            numChanged = numChanged + examineExample(smo,i1);
                        end
                    end
                end

                if(examineAll == 1)
                    examineAll = 0;
                elseif(examineAll == 0)
                    examineAll = 1;
                end
                
            end
            
            % Round LMs too close to 0 (numerical imprecision due to the error cache)
            for k=1:smo.N
                if smo.alpha(k) < 1e-10
                    smo.alpha(k) = 0;
                end
            end
            
            smo.isSupportVector = smo.alpha > 0;
            
            %Calculate the final bias of the model.
            % For numerical stability average over all support vectors, to
            % simplify the code average over all alpha (inefficient).
            bias = zeros(smo.N,1);
            for i=1:smo.N
                res = zeros(smo.N,1);
                for k=1:smo.N
                    res(k) = smo.kernel(smo.x(i,:),smo.x(k,:));
                end
                bias(i) = smo.y(i) - sum( smo.y .* smo.alpha .* res);
            end
            smo.bias = mean(bias);
            
            
        end
        
        function output = predict(smo,data)
            
            n = size(data);
            output = zeros(n(1),1);
            
            if(n(2) ~= size(smo.x,2))
                disp("data provided for evaluation and for training differ in dimension!");
                return;
            end
            
            for i=1:n(1)
                
                res = zeros(smo.N,1);
                for k=1:smo.N
                    
                    % Calculate the kernel only if the associated LM is greater than 0
                    if smo.alpha(k) > 0
                        res(k) = smo.kernel(smo.x(k,:),data(i,:));
                    end
                    
                end
                
                output(i) = sum(smo.alpha .* smo.y .* res) + smo.bias;
                
            end
            
        end
         
    end
end