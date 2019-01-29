classdef Jsmo < handle
    %jsmo class 
    %
    %   Implemets the Sequential Minimal Optimization algorithm in the
    %   version suggested by Joachims in the paper "Making Large-Scale SVM
    %   Learning Practical".
    %   An online free version of the paper is available at:
    %   http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf
    %   
    %
    %   PROPERTIES:
    %
    %   x = train set
    %   y = class label associated with the train set
    %   N = size of the train set
    %   alpha = vector of solution
    %   G = Gradient vector
    %   b = bias
    %   q = size of the Working set B
    %   C = determines the tradeoff between increasing the margin-size and 
    %       ensuring that the x lie on the correct side of the margin.
    %   orderedGradList = list contaning the index of the elements of the
    %                     dataset sort in descending order of:= Yi*grad(i)
    %   errorCache = cache containing prediction error for every training
    %                sample
    %   tolerance = (default 1e-3) tolerance in the strenght the KKT 
    %               conditions are fullfil.
    %   eps = (default 10e-5) treshold that has to be reached for an update
    %         (in the function takeStep) to be valid. [alphaNew - alphaOld > eps]
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
    %   Jsmo = constructor; the folllowing values are required as input parameter:
    %          -data 
    %          -classLabels 
    %          -C 
    %          -q 
    %          As optional parameter it can also take the tolerance allowded in the
    %          violation of the KKT condition (tolerance) and the maximum number of 
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
    %   checkOpt = checks if the values assigned so far to the LMs satisfay
    %              the KKT conditions within a degree of violation specified
    %              by the value tolerance.
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
        q;
        C;
        orderedGradList;
        
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
        
        function obj = Jsmo(data, classLabels, C, q, tolerance, maxiter)
            % JSMO Constructor
            
            % Checking optional parameter
            if nargin < 4
                disp("ERROR: Not enough input arguments, not possible to instatitate Jsmo");
                return;
            end
            
            if nargin > 4
                obj.tolerance = tolerance;
            end
            
            if nargin > 5
                obj.maxiter = maxiter;
            end
            
            obj.x = data;
            obj.y = classLabels;
            obj.C = C;
            
            % The size of the working set must be an even number, if it is
            % not the case automatically set it to be 2.
            if mod(q,2) ~= 0
                disp("ERROR: q must be even (q set to 2)");
                q = 2;
            end
            
            obj.q = q; %size of the working set
            obj.N = size(classLabels,1);
            
            % Initialize all Lagrange multipliers (LMs) to 0
            obj.alpha = zeros(obj.N,1);
            % Initialize the gradient w.r.t. all alphas equal to -1
            obj.G = -ones(obj.N,1);
            % initialize threshold to zero
            obj.bias = 0;
            % sort in descending order of Yi*grad(i), the indexes of the dataset
            [~,obj.orderedGradList] = sort(obj.y .* obj.G,'descend');
            
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
        
        function opt = checkOpt(smo)
            
            opt = 0;
            
            b_up = inf;
            b_down = -inf;
            for k=1:smo.N
                
                F = smo.y(k) .* smo.G(k);
                
                %if k is in I0 or I1 or I2
                if (smo.y(k) == 1 && smo.alpha(k) < smo.C) || (smo.y(k) == -1 && smo.alpha(k) > 0)
                    
                    if F < b_up
                        b_up = F;
                        i_up = k;
                    end
                end
                
                % if k is in I0 or I3 or I4
                if (smo.y(k) == -1 && smo.alpha(k) < smo.C) || (smo.y(k) == 1 && smo.alpha(k) > 0)

                    if F > b_down
                        b_down = F;
                        i_down = k;
                    end
                end
            end
            
            smo.violation(smo.iter+1) = b_down - b_up;
            if b_down < b_up + smo.tolerance
                opt = 1;
            end
        end
        
        function [check,B] = WorkingSetSelection(smo)
            
            B = zeros(smo.N,1);
            qSelected = 0;
            qSelTop = [];
            qSelDown = [];
            
            % Indexes that travers the sorted list of LMs
            TopIndex = 1;
            DownIndex = smo.N;
            
            %Main loop that select the indexes of the elements that compose the working set
            while(DownIndex - TopIndex > 2) && qSelected < smo.q
                
                i_top = smo.orderedGradList(TopIndex,:);
                i_down = smo.orderedGradList(DownIndex,:);
                
                %Pick element from the top of the list
                if (smo.y(i_top) == -1 && smo.alpha(i_top) < smo.C) || (smo.y(i_top) == 1 && smo.alpha(i_top) > 0)
                    if size(qSelTop,2) < smo.q/2
                        qSelTop = [qSelTop i_top];
                    end
                    
                end
                
                %Pick element from the bottom of the list
                if (smo.y(i_down) == 1 && smo.alpha(i_down) < smo.C) || (smo.y(i_down) == -1 && smo.alpha(i_down) > 0)
                    if size(qSelDown,2) < smo.q/2
                        qSelDown = [qSelDown i_down];
                    end
                end
                
                qSelected = size(qSelDown,2) + size(qSelTop,2);
                
                TopIndex = TopIndex + 1;
                DownIndex = DownIndex - 1;
                
            end
            
            % if it is not possible to return a working set composed of an
            % even number of elements coming from the top and bottom of the
            % list, round the working set in a way that the number of LMs coming
            % from the top is the same that the number coming from the
            % bottom.
            if size(qSelDown,2) > size(qSelTop,2)
                qSelDown = qSelDown(1,1:size(qSelTop,2));
            elseif size(qSelTop,2) > size(qSelDown,2)
                qSelTop = qSelTop(1,1:size(qSelDown,2));
            end
            
            % set the working set with both elements choosen from the top and
            % the bottom of the ordered list.
            for k=1:size(qSelTop,2)
                index = qSelTop(1,k);
                B(index) = 1;
            end
            
            for k=1:size(qSelDown,2)
                index = qSelDown(1,k);
                B(index) = 1;
            end
            
            % if the size of the working set is zero, return 0 and signal
            % that the algorithm has already reached convergence.
            check = size(qSelDown,2) + size(qSelTop,2);
        end
        
        function train(smo)
           
            while(smo.iter < smo.maxiter)
                
                smo.iter = smo.iter + 1;
                [check,B] = WorkingSetSelection(smo);
                
                % In case is not possible to select an even number > 0 of
                % LM (to build the working set) exit from the loop.
                if check == 0
                    break;
                end
                
%---- Calculate parameter involved in the quadratic optimization step -----
                
                [indexB,~] = find(B); % Active set(B)
                [indexN,~] = find(~B); % NonActive set(N)
                alphaN = smo.alpha(indexN);
                oldAlphaB = smo.alpha(indexB);
                yB = smo.y(indexB);
                yN = smo.y(indexN);
                
                Qbb = zeros(size(indexB,1),size(indexB,1));
                for k1=1:size(indexB,1)
                    i = indexB(k1);
                    for k2=1:size(indexB,1)
                        j = indexB(k2);
                        Qbb(k1,k2) = yB(k1)*yB(k2)*smo.kernel(smo.x(i,:),smo.x(j,:));
                    end
                end
                
                Qbn = zeros(size(indexB,1),size(indexN,1));
                for k1=1:size(indexB,1)
                    i = indexB(k1);
                    for k2=1:size(indexN,1)
                        j = indexN(k2);
                        Qbn(k1,k2) = yB(k1)*yN(k2)*smo.kernel(smo.x(i,:),smo.x(j,:));
                    end
                end
                
                f = -(ones(size(indexB,1),1) - Qbn*alphaN);
                Aeq = yB';
                beq = -alphaN' * yN;
                lowB = zeros(size(indexB,1),1); %Lower bound for LMs
                upB = smo.C .* ones(size(indexB,1),1); %Upper bound for LMs
                
                % Invoke quadratic solver (quadprog) and set the new alphas obtained
                opts = optimset('display','off');
                smo.alpha(indexB) = quadprog(Qbb,f,[],[],Aeq,beq,lowB,upB,[],opts);
                
                % Round non-boundary LMs too close to the boundaries, due
                % to the numerical tolerance of the quadprog solver.
                for k2=1:size(indexB,1)
                    k1 = indexB(k2);
                    if smo.alpha(k1) < 1e-7
                        smo.alpha(k1) = 0;
                    elseif smo.alpha(k1) > (smo.C - 1e-7)
                        smo.alpha(k1) = smo.C;
                    end
                end
                
% ----------------- Update the gradient -----------------------------------
                
                % For each variable in the Working set calculate how
                % much LM increase or decrease due to the optimization
                % step (deltaAlpha)
                deltaAlpha = zeros(size(indexB,1),1);
                for k1=1:size(indexB,1)
                    k2 = indexB(k1);
                    deltaAlpha(k1,1) = smo.alpha(k2) - oldAlphaB(k1);
                end             
                
                % Update the gradient for the elements not belonging to the 
                % working set using the kernel already computed for Qbn
                for k1=1:size(indexN,1)
                    i = indexN(k1);
                    for k2=1:size(indexB,1)
                        smo.G(i) = smo.G(i) + Qbn(k2,k1) * deltaAlpha(k2,1);
                    end
                end
                
                % Update the gradient for the elements of the working set
                % using the kernel already computed for Qbb
                for k1=1:size(indexB,1)
                    i = indexB(k1);
                    for k2=1:size(indexB,1)
                        smo.G(i) = smo.G(i) + Qbb(k2,k1) * deltaAlpha(k2,1);
                    end
                end
                
                %------------- Reorder the list with the updated gradient -----------------
                [~,smo.orderedGradList] = sort(smo.y .* smo.G,'descend');
                
                if smo.checkOpt()
                    break;
                end
                
            end
            
            % Round LMs too close to 0 (numerical imprecision)
            for k2=1:smo.N
                if smo.alpha(k2) < 1e-10
                    smo.alpha(k2) = 0;
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

