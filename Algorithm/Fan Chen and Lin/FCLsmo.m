classdef FCLsmo < handle
    %FCLsmo class 
    %
    %   Implemets the Sequential Minimal Optimization algorithm in the
    %   version suggested by Fan, Chin and Lin in the paper "Working set
    %   selection using 2nd order information for traning support vector
    %   machine".
    %   An online free version of the paper is available at:
    %   www.jmlr.org/papers/volume6/fan05a/fan05a.pdf
    %   
    %   The authors propose a new Working set selection procedure that
    %   exploit 2nd order information of the objective function.
    %   The paper begin with the analysis of the previously WSS procedure 
    %   used to train SVM by Sequential Minimal Optimization, mainly focusing 
    %   on the work of (Keerthi et al. 2001) which presents the Maximal Violating 
    %   Pair selection procedure.
    %   
    %   Starting from the fact that this procedure is related to the first 
    %   order approximation of the objective function, and the
    %   awareness that using 2nd order information usually help to achive
    %   better results, they proposed a new method to select the working
    %   set that work with 2nd order approximation.
    %   
    %   Since there is no way to efficently solved the 2nd order
    %   approximation problem, without checking all the (n 2) possible
    %   selection of couple of alphas, the authors proposed to
    %   only heuristically check several alphas. 
    %  
    %   As in the Maximal violating pair the first alpha (call it i) is such that:
    %   (0 < alpha(i) < C) or (alpha(i) = 0 and y(i) = 1) or (alpha(i) = C and y(i) = -1)
    %   and it maximize the quantity: F*y(i) (where F = w(alpha)*x(i) - y(i)).
    %   The second alpha (j) is such that the couple (i,j) maximized the
    %   2nd order approximation problem.
    %   Moreover, if (the hessian) H < 0 we have a way to efficently solved the
    %   problem of selecting the second alpha.
    %   On the other hand, if H >= 0 the objective function is modified
    %   with an additional term that make it strictly convex (this last 
    %   consideration is based on the work of Chen et al. 2006).
    %
    %   In the end of the paper the authors prove that checking all the (n 2) 
    %   couple does not reduce iterations much in respect of using the
    %   heuristic proposed.
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
    %   C = determines the tradeoff between increasing the margin-size and 
    %       ensuring that the x lie on the correct side of the margin.
    %   tolerance = (default 1e-3) tolerance in the strenght thee KKT
    %               condition are fullfil.
    %   iter = number of iterations the training last.
    %   maxiter = (default 10000) maximum number of iteration that the training
    %              algorithm is allow to run.
    %   violation = collects the degree of violation of the KKT conditions for
    %               each iteration.
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
    %
    %
    %   METHODS:
    %
    %   FCLsmo = constructor; the folllowing values are required as input parameter:
    %          -data 
    %          -classLabels 
    %          -C 
    %          As optional parameters it can also take the tolerance allowded in the
    %          violation of the KKT condition (tolerance), the value of the tau
    %          used to enforced convexity (tau) and the maximum number of 
    %          iterations that the algorithm can loops (maxiter)
    %
    %   kernel = It returns the value of the kernel K(Xi,Xj)
    %
    %   setKernel = It sets the type of kernel. take as required input the type
    %               of kernel ('gaussian', 'polynomial' and 'linear'). If
    %               the choice of the kernel is either polynomial or
    %               gaussian, is also possible to specify an optional
    %               parameter that modify rispectively the deegre of the
    %               polynomial kernel or the sigma of the gaussian kernel.
    %
    %   WorkingSetSelection = it retuns the indexes of the LMs that compose
    %                         the working set during the current iteration.
    %
    %   train = train the SVM
    %
    %
    %   predict = given a set of data points as input returns the prediction
    %             using the model generated.
    
    
    properties
        x;
        y;
        N;
        alpha;
        G;
        bias;
        tau = 1e-12;
        C;
        
        tolerance = 1e-3;
        iter = 0;
        maxiter = 10000;
        violation;
        
        kernelType = 'linear';
        degree = 2;
        sigma = 1;
        
        isSupportVector;
        kernelEvaluation = 0;
    end
    
    methods
        
        function obj = FCLsmo(data, classLabels, C, tolerance, tau, maxiter)
            % FCLSMO Constructor
            
            % Checking optional parameter
            if nargin < 3
                disp("ERROR: Not enough input arguments, not possible to instatitate FCLsmo");
                return;
            end
            
            if nargin > 3
                obj.tolerance = tolerance;
            end
            
            if nargin > 4
                obj.tau = tau;
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
            % Initialize the gradient w.r.t. all alphas equal to -1
            obj.G = -ones(obj.N,1);
            % initialize threshold to zero
            obj.bias = 0;
            
            obj.isSupportVector = zeros(obj.N,1);
            obj.violation = zeros(obj.maxiter,1);
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
        
        function [i,j] = WorkingSetSelection(smo)
            
            % Select first alpha (i)
            i = -1;
            G_max = -inf;
            G_min = inf;
            
            for k=1:smo.N
                % We look for the first alpha only among those indexes inside I(up)
                if(smo.y(k) == 1 && smo.alpha(k) < smo.C) || (smo.y(k) == -1 && smo.alpha(k) > 0)
                    
                    %Select the index i such that the value -y(k) * G(k) is maximized
                    if( -smo.y(k) * smo.G(k) >= G_max)
                        i = k;
                        G_max = - smo.y(k) * smo.G(k);
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
            smo.violation(smo.iter+1) = G_max - G_min; 
           
            if(G_max - G_min < smo.tolerance)
                i = -1;
                j = -1;
            end
        end
 
        function train(smo)
            
            while (smo.iter < smo.maxiter)
                
                [i,j] = WorkingSetSelection(smo);
                
                if(j == -1)
                    break;
                end
                
                a1 = smo.kernel(smo.x(i,:),smo.x(i,:));
                a2 = smo.kernel(smo.x(j,:),smo.x(j,:));
                a3 = smo.y(i) * smo.y(j) * smo.kernel(smo.x(i,:),smo.x(j,:));
                
                a = a1 + a2 - 2*a3;
                
                if a <= 0
                    a = smo.tau;
                end
                
                b = - smo.y(i) * smo.G(i) + smo.y(j) * smo.G(j);
                
                oldAlphaI = smo.alpha(i);
                oldAlphaJ = smo.alpha(j);
                smo.alpha(i) = smo.alpha(i) + smo.y(i)*(b/a);
                smo.alpha(j) = smo.alpha(j) - smo.y(j)*(b/a);
                
                
                constantSum = smo.y(i) * oldAlphaI + smo.y(j) * oldAlphaJ;
                
                if smo.alpha(i) > smo.C
                    smo.alpha(i) = smo.C;
                elseif smo.alpha(i) < 0
                    smo.alpha(i) = 0;
                end
                
                smo.alpha(j) = smo.y(j)*(constantSum - smo.y(i) * smo.alpha(i));
                
                if smo.alpha(j) > smo.C
                    smo.alpha(j) = smo.C;
                elseif smo.alpha(j) < 0
                    smo.alpha(j) = 0;
                end
                
                smo.alpha(i) = smo.y(i)*(constantSum - smo.y(j) * smo.alpha(j));
                
                % Update the gradient
                deltaAlphaI = smo.alpha(i) - oldAlphaI;
                deltaAlphaJ = smo.alpha(j) - oldAlphaJ;
                
                for k=1:smo.N
                    smo.G(k) = smo.G(k) + (smo.y(i) * smo.y(k) * smo.kernel(smo.x(i,:),smo.x(k,:)) * deltaAlphaI) + (smo.y(j) * smo.y(k) * smo.kernel(smo.x(j,:),smo.x(k,:)) * deltaAlphaJ);
                end
                
                smo.iter = smo.iter + 1;
            end
            
            % Round LMs too close to 0 (numerical imprecision)
            for k=1:smo.N
                if smo.alpha(k) < 1e-10
                    smo.alpha(k) = 0;
                end
            end
            
            smo.isSupportVector = smo.alpha > 0;
            smo.violation = smo.violation(1:smo.iter,1);
            
            %Calculate the final bias of the model.
            % For numerical stability average over all support vectors, to
            Allbias = zeros(smo.N,1);
            for i=1:smo.N
                res = zeros(smo.N,1);
                for k=1:smo.N
                    if smo.alpha(k) > 0
                        res(k) = smo.kernel(smo.x(i,:),smo.x(k,:));
                    end
                end
                Allbias(i) = smo.y(i) - sum( smo.y .* smo.alpha .* res);
            end
            smo.bias = mean(Allbias);
            
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
                
                if output(i) > 0
                    output(i) = 1;
                elseif output(i) < 0
                    output(i) = -1;
                end
                
            end
            
        end
    end
end

