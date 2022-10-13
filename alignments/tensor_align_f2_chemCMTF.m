function [alignment_matrix, initial_permutation, ACC_Pi, Loss_obj, Loss_pi] = tensor_align_f2_chemCMTF(Z, MaxMostOuterIter, thresholdResObj, lambda, str_factor, str_pi, k_cluster, guessRank, truePermu, str_row, str_col, str_entropy)
    % alignment_matrix: learned permutation matrix
    % truePermu: true permutation matrix
    % initial_permutation: the initialization pi matrix

    %% preprocessing factor A for accuracy of Pi
    % Cidx = kmeans(trueA, k_cluster);

    %% Get input data: tensor X and the tensor Y
    X = Z.object{1};
    Y = Z.object{2};

    % size of two input tensors
    szX = size(X);
    szY = size(Y);
    sz = [szX szY];
    mode1_size = sz(1);

    %% Hyper-parameters
    % MaxMostOuterIter = 500;
    resObj = 1e10;
    % thresholdResObj = 1e-6;

    % Ay hyper-parameters
    MaxAyIter = 10000;
    % lambda = 50 %800 % control how close two spaces are
    alpha_y = 0.00001; % for calculating Ay by gradient descent
    thresholdResidualAy = 1e-6; % residuals of updating Ay

    % Ax-Pi*Ay hyper-parameters
    lambda1 = 0.1;
    lambda2 = 0.1;
    lambda3 = 0.1;
    x0_init = 0.05;

    %% Initialize the CMTF rank R of tensor X and Y
    % trueRank = length(lambdas_data{1}) % hyper-parameter, need to guess rank of the tensor????
    cmtfRank = guessRank % trueRank
    fprintf(1, 'CMTF rank =%d \n', guessRank);
    initRank = guessRank; % the value of initRank starting from mode1_size
    fprintf(1, 'Initial rank =%d \n', initRank);


    %% Initialize permutation matrix
    if strcmp(str_pi, 'rand') % random decimal initialization
        initPermu = rand(mode1_size,mode1_size);
    elseif strcmp(str_pi, 'randPi') % random permutation initialization
        initPermu = eye(mode1_size,mode1_size);
        initPermu = initPermu(randperm(mode1_size),:); % this is the init permutation matrix \Pi
    elseif strcmp(str_pi, 'smart1') 
        [initPermu] = Initialization_1(X, Y);
    elseif strcmp(str_pi, 'smart2') 
        [initPermu] = Initialization_2(X, Y, initRank); % best initialization of \Pi
    elseif strcmp(str_pi, 'smart3') 
        [initPermu] = Initialization_3(X, Y);
    end
    initial_permutation = initPermu; % store the initial permutation matrix 

    %% Initialize A, B, C, D, E by CPD or Random
    if strcmp(str_factor, 'CPD') % doing cpd initialization
        % Initialize A, B, C, D, E by CPD
        [Fac] = cp_als(X, cmtfRank,'tol',10^-9,'maxiters',1e3);
        Ax = Fac.U{1};
        B = Fac.U{2};
        C = Fac.U{3};
        Y_1modeproduct = ttm(Y, initPermu, 1);
        [Fac2] = cp_als(Y_1modeproduct, cmtfRank,'tol',10^-9,'maxiters',1e3);
        Ay = Fac2.U{1}; % try CDP of Y*Pi
        % A = Fac2.U{1}; % try CDP of Y*Pi
        D = Fac2.U{2};
        E = Fac2.U{3};
    elseif strcmp(str_factor, 'rand') % doing cpd initialization
        % Initialize Ax, B, C, Ay, D, E by random
        Ax = rand(sz(1), guessRank);
        B = rand(sz(2), guessRank);
        C = rand(sz(3), guessRank);
        Ay = rand(sz(4), guessRank);
        D = rand(sz(5), guessRank);
        E = rand(sz(6), guessRank);
    end

    %% preprocessing factor A for accuracy of Pi
    Cidx = kmeans(Ax, k_cluster);

    %% function values
    func_val(1) = 0;
    Fac_val(1) = 0;
    Pi_val(1) = 0;
    accuracy_pi(1) = 0;

    %% --------------  Here should be in the MostOuterIter  -------------- %
    fprintf(1, 'Run our algorithm...\n');
    iter = 1;

    while(iter <= MaxMostOuterIter)
        %% 1. update Ax: 
        Ux = {Ax, B, C};
        X1ktrCB = mttkrp(X, Ux, 1); % X1*khatrirao(C, B)
        % optimization: X1*(C kr B) is too expensive, find matlab implementation of MTTKRP (matrixize tensor times kr product)
        CTC_times_BTB = ((C.')*C) .* ((B.')*B);
        [sizeCTCBTB, ~] = size(CTC_times_BTB);
        IdCB = eye(sizeCTCBTB, sizeCTCBTB);
        CB = CTC_times_BTB + lambda*IdCB; % mrdivide, /; x = B/A
        Ax = (X1ktrCB + lambda * initPermu * Ay) / (CB + 10^(-7)*eye(sizeCTCBTB));

        %% 2. update B
        Ux = {Ax, B, C};
        X2ktrCAx = mttkrp(X, Ux, 2); % X2*khatrirao(C, Ax)
        CTC_times_AxTAx = ((C.')*C) .* ((Ax.')*Ax);
        CAx = CTC_times_AxTAx;
        [sizeCTCAxTAx, ~] = size(CTC_times_AxTAx);
        B = X2ktrCAx / (CAx + 10^(-7)*eye(sizeCTCAxTAx));

        %% 3. update C
        Ux = {Ax, B, C};
        X3ktrBAx = mttkrp(X, Ux, 3); % X3*khatrirao(B, Ax)
        BTB_times_AxTAx = ((B.')*B) .* ((Ax.')*Ax);
        BAx = BTB_times_AxTAx;
        [sizeBTBAxTAx, ~] = size(BTB_times_AxTAx);
        C = X3ktrBAx / (BAx + 10^(-7)*eye(sizeBTBAxTAx));

        %% 4. update Ay by Gradient Descent
        Uy = {Ay, D, E};
        Y1ktrED = mttkrp(Y, Uy, 1); % Y1*khatrirao(E, D)
        ETE_times_DTD = ((E.')*E) .* ((D.')*D);
        lamPiT = lambda*(initPermu.');
        lamPiTPi = lambda*(initPermu.')*initPermu;
        itAy = 1;
        residualAy = 1e10;
        while (itAy <= MaxAyIter || residualAy >= thresholdResidualAy) 
            gdAy = -Y1ktrED + Ay * ETE_times_DTD + lamPiTPi*Ay - lamPiT*Ax;
            Ay_prev = Ay;
            Ay = Ay - alpha_y*gdAy; % update Ay here by GD
            residualAy = norm(Ay - Ay_prev, "fro")^2;
            itAy = itAy+1;
        end
        Ay

        %% 5. update D
        Uy = {Ay, D, E};
        Y2ktrEAy = mttkrp(Y, Uy, 2); % Y2*khatrirao(E, Ay)
        ETE_times_AyTAy = ((E.')*E) .* ((Ay.')*Ay);
        EAy = ETE_times_AyTAy;
        [sizeETEAyTAy, ~] = size(ETE_times_AyTAy);
        D = Y2ktrEAy / (EAy + 10^(-5)*eye(sizeETEAyTAy));

        %% 6. update E
        Uy = {Ay, D, E};
        Y3ktrDAy = mttkrp(Y, Uy, 3); % Y3*khatrirao(D, Ay)
        % ktrDAy = khatrirao(D, Ay);
        DTD_times_AyTAy = ((D.')*D) .* ((Ay.')*Ay);
        DAy = DTD_times_AyTAy;
        [sizeDTDAyTAy, ~] = size(DTD_times_AyTAy);
        E = Y3ktrDAy / (DAy + 10^(-5)*eye(sizeDTDAyTAy));

        %% 7. update Pi by solver
        %% Learn permutation matrix by solving func with non-linear constraints
        [alignment_matrix] = LS_nonlcon_solver(Ax, Ay, lambda1, lambda2, lambda3, x0_init);
        % normalize initPermu to sumRow(initPermu)=1
        % sumRowPi = sum(alignment_matrix, 2); % sum of each row
        % sumRowPi = sumRowPi.';
        % alignment_matrix = bsxfun(@rdivide, alignment_matrix, sumRowPi(:)); % divide each row of the matrix with corresponding element in row vector sumRow
        
        % Update initial permutation matrix: initPermu
        imagesc(alignment_matrix);
        initPermu = alignment_matrix;
        alignment_matrix


        %% 9. update loss here
        X_new = ktensor({Ax, B, C});
        X_new = tensor(X_new);
        f_X = norm(X - X_new)^2;
        Y_new = ktensor({Ay, D, E});
        Y_new = tensor(Y_new);
        f_Y = norm(Y - Y_new)^2;
        %% Calculate the loss of ||Ax - \Pi*Ay||fro
        f_factors = norm(Ax - alignment_matrix * Ay, "fro")^2; 
        %% Calculate the loss of alignment_matrix and the true permutation matrix
        f_pi = norm(alignment_matrix - truePermu, "fro")^2;
        %% evaluate the obj function: min ||X - [Ax, B, C]||_fro^2 + ||Y - [Ay, D, E]||_fro^2 + ||Ax - \Pi*Ay||_fro^2
        f_objloss = f_X + f_Y + lambda*f_factors;

        acc_pi = 0;
        for i = 1:mode1_size
            for j = 1:mode1_size
                if truePermu(i,j)==1 && alignment_matrix(i,j)>=0.4
                    acc_pi = acc_pi+1;
                end
            end
        end
        acc_pi = acc_pi/mode1_size;


        if iter==1
            fprintf(1,' MostOuterIter  f loss func    f factors      f permutation       Pi Accuracy  \n');
            fprintf(1,'--------------  -------------  -------------  ---------------   ---------------\n');
        end
        fprintf(1,'%7d %16f %16f %16f %16f %16f %16f\n', iter, f_objloss, f_factors, f_pi, acc_pi);

        func_val(iter) = f_objloss;
        Fac_val(iter) = f_factors;
        Pi_val(iter) = f_pi;
        accuracy_pi(iter) = acc_pi;


        %% Residual of obj
        if iter >= 2
            resObj = abs(f_objloss - func_val(iter-1)) / f_objloss;
        end
        if resObj <= thresholdResObj
            fprintf(1, 'Converged! \n');
            break;
        end

        iter = iter+1;
    end

    % output these loss values
    Res = cell(4,1);
    Res{1} = func_val;
    Res{2} = Fac_val;
    Res{3} = Pi_val;
    Res{4} = accuracy_pi;


    % final Loss
    Loss_obj = f_objloss;
    Loss_pi = f_pi;

    % calculating the accuracy of the learned Pi
    ACC_Pi = acc_pi
    kmeans_acc_pi = 0;
    for i = 1:mode1_size
        for j = 1:mode1_size
            if truePermu(i,j)==1 && alignment_matrix(i,j)>=0.4
                kmeans_acc_pi = kmeans_acc_pi+1;
            elseif truePermu(i,j)==0 %
                % if row i and j are in same cluster: Cidx(i)==Cidx(j), then count it as true
                % otherwise row i and j are not in same cluster, j cannot change to i-th position!!
                if alignment_matrix(i,j)>=0.4 && Cidx(i)==Cidx(j)
                    kmeans_acc_pi = kmeans_acc_pi+1;
                end
            end
        end
    end
    kmeans_acc_pi = kmeans_acc_pi/mode1_size;
    ACC_Pi = kmeans_acc_pi

end