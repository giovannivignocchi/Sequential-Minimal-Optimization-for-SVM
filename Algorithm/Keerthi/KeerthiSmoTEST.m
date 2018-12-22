classdef KeerthiSmoTEST < handle
    %KeerthiSmo class 
    %
    %   Implemets the Sequential Minimal Optimization algorithm in the
    %   version suggested by Keerthi in the paper "Improvements to Platt's SMO 
    %   algorithm for SVM Classifier Design".
    %   An online free version of the paper is available at:
    %   http://web.cs.iastate.edu/~honavar/keerthi-svm.pdf
    %
    %
    %   PROPERTIES:
    %
    %   x = train set
    %   y = class label associated with the train set
    %   N = size of the train set
    %   alpha = vector of solution
    %   b = 
    %   Fcache =
    %   i_up =
    %   i_down =
    %   b_up =
    %   b_down =
    %   tolerance = (default 1e-3) tolerance in the strenght thee KKT
    %               condition are fullfil.
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
    %   alphaHistory = vector recording the behaviour of aplhas during the
    %                  iteration of the algorithm.
    %   kernelEvaluation = varable that records the number of kernel
    %                      evaluation carried out during the iteration of
    %                      the algorithm.    
    properties
        x;
        y;
        N;
        alpha;
        b;
        Fcache;
        i_up;
        i_down;
        b_up;
        b_down;
        C;
        
        tolerance = 1e-3;
        eps = 10e-5;
        iter = 0;
        maxiter = 200;
        
        kernelType = 'linear';
        degree = 2;
        sigma = 1;
        
        isSupportVector;
        alphaHistory;
        kernelEvaluation = 0;
    end
    
    methods
        
        function obj = KeerthiSmoTEST(data, classLabels, C, tolerance, eps, maxiter)
            % Keerthi SMO Constructor
            
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
            obj.b = 0;
            
            %Initialize b_up and b_down to assure violation is guaranteed
            obj.b_up = -1;
            obj.b_down = 1;
            % initializie i_up to the index of any element of class 1
            [~,obj.i_up] = max(obj.y == 1);
            % initializie i_down to the index of any element of class -1
            [~,obj.i_down] = max(obj.y == -1);
            
            %Initialize the Fcache
            obj.Fcache = zeros(obj.N,1);
            obj.Fcache(obj.i_up) = -1;
            obj.Fcache(obj.i_down) = 1;
            
            obj.isSupportVector = zeros(obj.N,1);
            obj.alphaHistory = zeros(obj.N,obj.maxiter);
        end
        
        function ker = kernel(smo,x1,x2)
            % Calculate the value of the selected type of kernel [K(x1,k2)]
            
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
        
        function [L,H] = calculateBoundaries(smo,i1,i2,alphaOld1,alphaOld2)
            % Return the High (H) and Low (L) margins between which the LMs that
            % are under joint optimization must lie.
            if (smo.y(i1) ~= smo.y(i2) )
                L = max(0, alphaOld2 - alphaOld1);
                H = min(smo.C, smo.C + alphaOld2 - alphaOld1);
            else
                L = max(0, alphaOld2 + alphaOld1 - smo.C);
                H = min(smo.C, alphaOld2 + alphaOld1);
            end
        end
        
        function Fi = calcFi(smo,i)
            % Calculate F for the element i.
            % Note that Fi = - grad(i) * y(i), this quantitiy will be used
            % both to check for optimality and select the next LM to be
            % optimized.
            
            res = zeros(smo.N,1);
            for k=1:smo.N
                res(k) = smo.kernel(smo.x(i,:),smo.x(k,:));
            end
            u = sum ( smo.alpha .* smo.y .* res );
            Fi = u - smo.y(i);
        end
        
        function update = examineExample(smo,i1)
            % Given the index of the first LM that compose the working set,
            % examineExample search for the second LM and perform the update.
            %
            % If the function succeed in updating a couple of LMs (i1, ix)
            % it returns 1, 0 otherwise.
            
            update = 0;

            if (smo.alpha(i1) > 0 && smo.alpha(i1) < smo.C)
                F1 = smo.Fcache(i1);
            else
                F1 = smo.calcFi(i1);
                smo.Fcache(i1) = F1;
                
                % Update (b_low, i_low) or (b_up, i_up) using (F1, i1)
                if (smo.alpha(i1) == 0 && smo.y(i1) == 1) || (smo.alpha(i1) == smo.C && smo.y(i1) == -1) && (F1 < smo.b_up)
                    
                    smo.b_up = F1;
                    smo.i_up = i1;
                    
                elseif (smo.alpha(i1) == smo.C && smo.y(i1) == 1) || (smo.alpha(i1) == 0 && smo.y(i1) == -1) && (F1 > smo.b_down)
                    
                    smo.b_down = F1;
                    smo.i_down = i1;
                    
                end
                
            end
            
            % Check optimality using current b_down and b_up,
            % If violated, find an index i2 to jointly optimize with i1
             optimality = 1;
             
             % if i1 is in Iup
             if (smo.y(i1) == 1 && smo.alpha(i1) < smo.C) || (smo.y(i1) == -1 && smo.alpha(i1) > 0)
                if smo.b_down - F1 > smo.tolerance
                    optimality = 0;
                    i2 = smo.i_down;
                end
             end
             
             % if i1 is in Idown
             if (smo.y(i1) == -1 && smo.alpha(i1) < smo.C) || (smo.y(i1) == 1 && smo.alpha(i1) > 0)
                 if F1 - smo.b_up > smo.tolerance
                    optimality = 0;
                    i2 = smo.i_up;
                 end
             end
             
             if optimality == 1
                 return
             end
             
             %If i1 is a non boundary example and b_up <= Fi1 <= b_down
             %both i_down and i_up are valid choice. Then we need to choose
             %the best among the 2 possible values.
             if (smo.alpha(i1) > 0 && smo.alpha(i1) < smo.C)
                 if smo.b_down - F1 > F1 - smo.b_up
                     i2 = smo.i_down;
                 else
                     i2 = smo.i_up;
                 end
             end  
             
             update = smo.takeStep(i1,i2);
             
        end
        
        function update = takeStep(smo,i1,i2)
            % This function carry out the joint optimization of the LMs at the indexes i1 and i2.
            % If the difference of the new value for the first LM with respect
            % to its previous value is sufficiently big, the update is carry
            % out and takeStep retutn 1, otherwise the update is refused and
            % return 0.
            
            update = 0;
            
            if (i1 == i2)
                return;
            end
            
            s = smo.y(i1) * smo.y(i2);
            
            if (smo.alpha(i1) > 0 && smo.alpha(i1) < smo.C)
                F1 = smo.Fcache(i1);
            else
                F1 = smo.calcFi(i1);
                smo.Fcache(i1) = F1;
            end
            
            if (smo.alpha(i2) > 0 && smo.alpha(i2) < smo.C)
                F2 = smo.Fcache(i2);
            else
                F2 = smo.calcFi(i2);
                smo.Fcache(i2) = F2;
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
                
                alphaNew2 = alphaOld2 + smo.y(i2) * (F2 - F1) / eta;
                
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
                c2 = smo.y(i2) * (F1-F2) - eta * alphaOld2;
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
            
            % Check if the new value for alphaNew2 is sufficently different
            % from the previous value (alphaOld2)
            if abs(alphaNew2-alphaOld2) < smo.eps
                return;
            end
            
            %After this point the old LMs will be updated to a1 and a2
            update = 1;
            
            %Since a1 + s * a2 = alph1 + s * alph2
            alphaNew1 = alphaOld1 + s * (alphaOld2 - alphaNew2);
            
            % Check if alphaNew1 is a legal value for a LM, if it is not
            % between 0 and C, update alphaNew2 accordingly
            if (alphaNew1 < 0)
                alphaNew2 = alphaNew2 + s * alphaNew1;
                alphaNew1 = 0;
            elseif (alphaNew1 > smo.C)
                t = alphaNew1-smo.C;
                alphaNew2 = alphaNew2 + s * t;
                alphaNew1 = smo.C;
            end
            
            % Round the new alphas that are too close to the boundaries.
            if alphaNew2 < 1e-7
                alphaNew2 = 0;
            elseif alphaNew2 > (smo.C - 1e-7)
                alphaNew2 = smo.C;
            end
            
            if alphaNew1 < 1e-7
                alphaNew1 = 0;
            elseif alphaNew1 > (smo.C - 1e-7)
                alphaNew1 = smo.C;
            end
            
            
            %Update the vector of LMs
            smo.alpha(i1) = alphaNew1;
            smo.alpha(i2) = alphaNew2;
            
            deltaAlpha1 = alphaNew1 - alphaOld1;
            deltaAlpha2 = alphaNew2 - alphaOld2;
            
            % Update Fcache for non boundary LMs
            for k=1:smo.N
                if (smo.alpha(k) > 0 && smo.alpha(k) < smo.C)
                    smo.Fcache(k) = smo.Fcache(k) + smo.y(i1) * deltaAlpha1 * smo.kernel(smo.x(i1,:),smo.x(k,:)) + smo.y(i2) * deltaAlpha2 * smo.kernel(smo.x(i2,:),smo.x(k,:));
                end
            end
            smo.Fcache(i1) = smo.Fcache(i1) + smo.y(i1) * deltaAlpha1 * smo.kernel(smo.x(i1,:),smo.x(i1,:)) + smo.y(i2) * deltaAlpha2 * smo.kernel(smo.x(i1,:),smo.x(i2,:));
            smo.Fcache(i2) = smo.Fcache(i2) + smo.y(i1) * deltaAlpha1 * smo.kernel(smo.x(i1,:),smo.x(i2,:)) + smo.y(i2) * deltaAlpha2 * smo.kernel(smo.x(i2,:),smo.x(i2,:));
            
            if (smo.alpha(i1) == 0 && smo.y(i1) == 1) || (smo.alpha(i1) == smo.C && smo.y(i1) == -1)
                if smo.Fcache(i1) < smo.b_up
                    smo.b_up = smo.Fcache(i1);
                    smo.i_up = i1;
                end
            elseif (smo.alpha(i1) == smo.C && smo.y(i1) == 1) || (smo.alpha(i1) == 0 && smo.y(i1) == -1)
                if smo.Fcache(i1) > smo.b_down
                    smo.b_down = smo.Fcache(i1);
                    smo.i_down = i1;
                end
            end
            
            if (smo.alpha(i2) == 0 && smo.y(i2) == 1) || (smo.alpha(i2) == smo.C && smo.y(i2) == -1)
                if smo.Fcache(i2) < smo.b_up
                    smo.b_up = smo.Fcache(i2);
                    smo.i_up = i2;
                end
            elseif (smo.alpha(i2) == smo.C && smo.y(i2) == 1) || (smo.alpha(i2) == 0 && smo.y(i2) == -1)
                if smo.Fcache(i2) > smo.b_down
                    smo.b_down = smo.Fcache(i2);
                    smo.i_down = i2;
                end
            end
            
            for k=1:smo.N
                if (smo.alpha(k) > 0 && smo.alpha(k) < smo.C)
                    
                    if smo.Fcache(k) < smo.b_up
                        smo.b_up = smo.Fcache(k);
                        smo.i_up = k;
                    end
                    
                    if smo.Fcache(k) > smo.b_down
                        smo.b_down = smo.Fcache(k);
                        smo.i_down = k;
                    end 
                    
                end
            end    
        end
        
        function train(smo)
            
            numChanged = 0;
            examineAll = 1;
            while (numChanged > 0 || examineAll) && smo.iter < smo.maxiter
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
                        if (smo.alpha(i1) > 0 && smo.alpha(i1) < smo.C)
                            numChanged = numChanged + examineExample(smo,i1);
                        end
                        
                        if smo.b_up > smo.b_down - smo.tolerance
                            break;
                        end
                    end
                end

                if(examineAll == 1)
                    examineAll = 0;
                elseif(examineAll == 0)
                    examineAll = 1;
                end
                smo.alphaHistory(:,smo.iter) = smo.alpha; 
                
            end
            
            % Round LMs too close to 0 (numerical imprecision due to the Fcache)
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
            smo.b = mean(bias);
            
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
                    res(k) = smo.kernel(smo.x(k,:),data(i,:));
                end
                
                output(i) = sum(smo.alpha .* smo.y .* res) + smo.b;
            end
            
        end
    end
end

