classdef KeerthiSmo < handle
    %Jsmo class 
    %
    %   Implemets the Sequential Minimal Optimization algorithm in the
    %   version suggested by Joachims in the paper "Making large-scale SVM
    %   learning practical".
    %   An online free version of the paper is available at:
    %   www.cs.cornell.edu/~tj/publications/joachims_99a.pdf
    %
    %
    %   PROPERTIES:
    %
    %   x = train set
    %   y = class label associated with the train set
    %   N = size of the train set
    %   alpha = vector of solution
    %   G = Gradient vector
    %   bias = bias of the solution provided.
    %   tau = (default 1e-12) value used when H >= 0 in order to make the
    %          objective function strictly convex.
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
    
    properties
        x;
        y;
        N;
        alpha;
        Fcache;
        i_up;
        i_down;
        b_up;
        b_down;
        C;
        
        tolerance = 1e-3;
        iter = 0;
        maxiter = 200;
        
        kernelType = 'linear';
        degree = 2;
        sigma = 1;
        
        isSupportVector;
        alphaHistory;
    end
    
    methods
        
        function obj = KeerthiSmo(data, classLabels, C, tolerance, maxiter)
            % SMO Constructor
            
            % Checking optional parameter
            if nargin < 3
                disp("ERROR: Not enough input arguments, not possible to instatitate FCLsmo");
                return;
            end
            
            if nargin > 3
                obj.tolerance = tolerance;
            end
            
            if nargin > 4
                obj.maxiter = maxiter;
            end
            
            obj.x = data;
            obj.y = classLabels;
            obj.C = C;
            obj.N = size(classLabels,1);
            
            % Initialize all Lagrange multipliers (LMs) to 0
            obj.alpha = zeros(obj.N,1);
            
            obj.b_up = -1;
            obj.b_down = 1;
            % initializie i_up to the index of any element of class 1
            [~,obj.i_up] = max(obj.y == 1);
            % initializie i_up to the index of any element of class -1
            [~,obj.i_up] = max(obj.y == -1);
            
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
        
        function [i,j] = WorkingSetSelection(smo)
            
            % Select first alpha (i)
            i = -1;
            G_max = -inf;
            G_min = inf;
            
            for k=1:smo.N
                % We look for the first alpha only among those indexes inside I(up)
                if(smo.y(k) == 1 && smo.alpha(k) < smo.C) || (smo.y(k) == -1 && smo.alpha(k) > 0)
                    
                    %Select the index i such that the value -y(k) * G(k) is maximized
                    if( -(smo.y(k)) * smo.G(k) >= G_max)
                        i = k;
                        G_max = -(smo.y(k)) * smo.G(k);
                    end
                    
                end
            end
            
            % Select the second alpha (j)
            j = -1;
            obj_min = inf;
            
            for k=1:smo.N
                
                % We look for the second alpha only among those indexes inside I(low)
                if(smo.y(k) == 1 && smo.alpha(k) > 0) || (smo.y(k) == -1 && smo.alpha(k) < smo.C)
                    
                    if( -smo.y(k) * smo.G(k) <= G_min)
                        G_min = -smo.y(k) * smo.G(k);
                    end
                    
                    % Theorem (3) of the paper provide an efficient way to
                    % evaluate the 2nd order approximation of th objective function.
                    % In order to do it we need to calculate a and b.
                    
                    b = G_max + (smo.y(k) * smo.G(k));
                    
                    %Otherwise we don't have a violating pair (i,j)
                    if(b > 0)
                        
                        a1 = smo.kernel(smo.x(i,:),smo.x(i,:));
                        a2 = smo.kernel(smo.x(k,:),smo.x(k,:));
                        a3 = smo.y(i) * smo.y(k) * smo.kernel(smo.x(i,:),smo.x(k,:));
                        
                        a = a1 + a2 - 2*a3;
                        
                        % if a<=0 the function is no more strictly convex
                        % and we need to modify it to enforce convexity.
                        if(a <= 0)
                            a = smo.tau;
                        end
                        
                        %We pick the one such that the couple (i,j) minimize
                        %the 2nd order approximation of the objective function
                        if(-(b*b) / a <= obj_min)
                            j = k;
                            obj_min = -(b*b) / a;
                        end
                    end
                end
            end
            
            % If the values of G_max e G_min are closed enough it
            % means that there are no (i,j) violating the kkt
            % condition and the solution found so far is optimal.
            if(G_max - G_min < smo.tolerance)
                i = -1;
                j = -1;
            end
        end
        
        function update = examineExample(smo,i1)
            % Given the index of the first LM that compose the working set,
            % examineExample search for the second LM and perform the update.
            % To optimized the selection of the second LM is used a
            % hyerarchy of heuristics.
            %
            % If the function succeed in updating a couple of LMs (i1, ix)
            % it returns 1, 0 otherwise.
            
            update = 0;

            if (smo.alpha(i1) > 0 && smo.alpha(i1) < smo.C)
                F1 = smo.Fcache(i1);
            else
                F1 = smo.calcFi;
                smo.Fcache(i1) = F1;
            end
            
            % Update (b_low, i_low) or (b_up, i_up) using (F1, i1)
            if (smo.alpha(i1) == 0 && smo.y(i1) == 1) || (smo.alpha(i1) == smo.C && smo.y(i1) == -1) && (F1 < smo.b_up)
               
               smo.b_up = F1;
               smo.i_up = i1;
               
            elseif (smo.alpha(i1) == smo.C && smo.y(i1) == 1) || (smo.alpha(i1) == 0 && smo.y(i1) == -1) && (F1 > smo.b_down)
                   
               smo.b_down = F1;
               smo.i_down = i1;
               
            end
            
            % Check optimality using current b_down and b_up,
            % If violated, find an index i2 to jointly optimize with i1
             optimality = 1;
             
             % if i1 is in Iup
             if (smo.y(i1) == 1 && smo.alpha(i1) < smo.C) || (smo.y(k) == -1 && smo.alpha(k) > 0)
                if smo.b_down - F1 > 2 * smo.tolerance
                    optimality = 0;
                    i2 = smo.i_down;
                end
             end
             
             % if i1 is in Idown
             if (smo.y(i1) == -1 && smo.alpha(i1) < smo.C) || (smo.y(k) == 1 && smo.alpha(k) > 0)
                 if F1 - smo.b_up > 2 * smo.tolerance
                    optimality = 0;
                    i2 = smo.i_up;
                 end
             end
             
             if optimality == 0
                 return
             end
             
             % if i1 is in Io choose the best i2
             if (smo.alpha(i1) > 0 && smo.alpha(i1) < smo.C)
                 if smo.b_down - F1 > 2 * smo.tolerance
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
            F1 = smo.calcFi(i1);
            F2 = smo.calcEi(i2);
            alphaOld1 = smo.alpha(i1);
            alphaOld2 = smo.alpha(i2);
            
            [Low, High] = smo.calculateBoundaries(i1,i2,alphaOld1,alphaOld2);
            
            if (Low == High)
                return;
            end
            
            k11 = smo.kernel(smo.x(i1,:),smo.x(i1,:))';
            k12 = smo.kernel(smo.x(i1,:),smo.x(i2,:))';
            k22 = smo.kernel(smo.x(i2,:),smo.x(i2,:))';
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
            
            %Update the vector of LMs
            smo.alpha(i1) = alphaNew1;
            smo.alpha(i2) = alphaNew2;
            
            
            
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
                        
                        if smo.b_up > smo.b_down - 2 * smo.tolerance
                            examineAll = 1;
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
                
                output(i) = sum(smo.alpha .* smo.y .* res) + smo.bias;
            end
            
        end
    end
end

