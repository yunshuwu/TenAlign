function [alignment_matrix, truePermu, initial_permutation, ACC_Pi, Loss_obj, Loss_pi] = tensor_align_f1_conTest(mode1_size, MaxMostOuterIter, thresholdResObj, str_factor, str_pi, k_cluster, trueRank, guessRank, str_row, str_col, str_entropy)
    % alignment_matrix: learned permutation matrix
    % truePermu: true permutation matrix
    % initial_permutation: the initialization pi matrix

    %% specify synthetic data
    % mode1_size = 10;
    sz     = [mode1_size 30 40 mode1_size 70 10]; %size of each mode
    P      = 2; %number of tensors
    lambdas_data= {ones(1,trueRank), ones(1,trueRank)}; % {[1 1 1 1 1], [1 1 1 1 1]}; % norms of components in each data set (length of each array specifies the number of components in each dataset)
    modes  = {[1 2 3], [4 5 6]}; % which modes belong to which dataset: every mode should have its unique number d, sz(d) corresponds to size of that mode
    noise = 0; %level of noise, for gaussian noise only!
    distr_data = {@(x,y) rand(x,y),@(x,y) randn(x,y),@(x,y) randn(x,y),@(x,y) rand(x,y),@(x,y) rand(x,y),@(x,y) randn(x,y)}; % function handle of distribution of data within each factor matrix /or Delta if linearly coupled, x,y are the size inputs %coupled modes need to have same distribution! If not, just the first one will be considered
    % distr_data = {@(x,y) randi(10,x,y),@(x,y) randi(10,x,y),@(x,y) randi(10,x,y),@(x,y) randi(10,x,y),@(x,y) randi(10,x,y),@(x,y) randi(10,x,y)};
    normalize_columns = 0; %wether or not to normalize columns of the created factor matrices, this might destroy the distribution!

    %% specify couplings
    rand('state',0);
    %% True couping: the probability distribution "permutation" matrix
    coupling.lin_coupled_modes = [1 0 0 1 0 0]; % which modes are coupled, coupled modes get the same number (0: uncoupled)
    coupling.coupling_type = [1]; % for each coupling number in the array lin_coupled_modes, set the coupling type: 0 exact coupling, 1: HC=Delta, 2: CH=Delta, 3: C=HDelta, 4: C=DeltaH
    coupling.coupl_trafo_matrices = cell(6,1); % cell array with coupling transformation matrices for each mode (if any, otherwise keep empty)

    coupling.coupl_trafo_matrices{1} = eye(mode1_size,mode1_size);

    truePermu = eye(mode1_size,mode1_size);
    truePermu = truePermu(randperm(mode1_size),:) % this is the true permutation matrix \Pi
    coupling.coupl_trafo_matrices{4} = truePermu;

    %% Fake coupling: coupling_pi
    coupling_pi.lin_coupled_modes = [1 0 0 1 0 0]; % which modes are coupled, coupled modes get the same number (0: uncoupled)
    coupling_pi.coupling_type = [1]; % for each coupling number in the array lin_coupled_modes, set the coupling type: 0 exact coupling, 1: HC=Delta, 2: CH=Delta, 3: C=HDelta, 4: C=DeltaH
    coupling_pi.coupl_trafo_matrices = cell(6,1); % cell array with coupling transformation matrices for each mode (if any, otherwise keep empty)

    coupling_pi.coupl_trafo_matrices{1} = eye(mode1_size,mode1_size); % both Pi_{1} and Pi_{4} are Identity matrix
    coupling_pi.coupl_trafo_matrices{4} = eye(mode1_size,mode1_size);

    %% set the fitting function for each dataset: 'Frobenius' for squared
    % Frobenius norm, 'KL' for KL divergence, IS for Itakura-Saito, 'beta' for other beta divergences (give beta in loss_function_param),...more todo
    loss_function{1} = 'Frobenius';
    loss_function{2} = 'Frobenius';
    loss_function_param{1} = [];
    loss_function_param{2} = [];
    %% check model
    check_data_input(sz,modes,lambdas_data,coupling);

    %% set initialization options
    init_options.lambdas_init = {ones(1,trueRank), ones(1,trueRank)}; % {[1 1 1 1 1], [1 1 1 1 1]}; %norms of components in each data set for initialization
    init_options.nvecs = 0; % wether or not to use cmtf_nvecs.m funcion for initialization of factor matrices Ci (if true, distr_data and normalize are ignored for Ci, not for Zi)
    init_options.distr = distr_data; % distribution of the initial factor matrices and their auxiliary variables
    init_options.normalize = 0; % wether or not to normalize the columns of the initial factor matrices (might destroy the distribution)

    %% set constraints
    constrained_modes = [1 0 0 1 1 0]; % 1 if the mode is constrained in some way, 0 otherwise, put the same for coupled modes!
    prox_operators = cell(6,1); % cell array of length number of modes containing the function handles of proximal operator for each mode, empty if no constraint

    prox_operators{1} = @(x,rho) project_box(x,0,inf); % non-negativity
    prox_operators{4} = @(x,rho) project_box(x,0,inf); % non-negativity
    prox_operators{5} = @(x,rho) project_box(x,0,inf); % non-negativity

    %% set weights
    weights = [1/2 1/2]; %weight w_i for each data set

    %% set lbfgsb options (only needed for loss functions other than Frobenius)
    lbfgsb_options.m = 6;%5; % should be number of mode?
    lbfgsb_options.printEvery = -1;
    lbfgsb_options.maxIts = 100;
    lbfgsb_options.maxTotalIts = 1000;
    lbfgsb_options.factr = 1e-6/eps;
    lbfgsb_options.pgtol = 1e-4;

    %% build model
    Z.loss_function = loss_function;
    Z.loss_function_param = loss_function_param;
    Z.modes = modes;
    Z.size  = sz;
    Z.coupling = coupling_pi; % use two identity matrix: Pi{1}=I, Pi{4}=I
    Z.constrained_modes = constrained_modes;
    Z.prox_operators = prox_operators;
    Z.weights = weights;

    %% create data
    % X: 存生成的所有tensor, 而且返回的X{}的确是tensor类型！
    % Atrue: 存用于生成tensor的factors
    [X, Atrue, Deltatrue,sigmatrue] = create_coupled_data('size', sz, 'modes', modes, 'lambdas', lambdas_data, 'noise', noise,'coupling',coupling,'normalize_columns',normalize_columns,'distr_data',distr_data,'loss_functions',Z.loss_function); %create data
    %% create Z.object 
    for p=1:P
        Z.object{p} = X{p}; % Z.object{} is also tensor!
        normZ{p} = norm(Z.object{p}); % norm(X) is the largest singular value of X
        Z.object{p} = Z.object{p}/normZ{p};
    end

    % print X type
    fprintf(1,'print tensor X typ... \n');
    class(Z.object{1})

    % fprintf(1, 'this is the factor A \n');
    trueA = Atrue{1};
    trueB = Atrue{2};
    trueC = Atrue{3};
    trueD = Atrue{5};
    trueE = Atrue{6};

    %% only for testing
    for p=1:P
        true_ktensor{p} =(ktensor(lambdas_data{p}'./normZ{p}, Atrue(modes{p})));
    end

    %% preprocessing factor A for accuracy of Pi
    % Cidx = kmeans(trueA, k_cluster);

    %% Get input data: tensor X and the tensor Y
    X = Z.object{1};
    Y = Z.object{2};

    %% Hyper-parameters
    % MaxMostOuterIter = 500;
    resObj = 1e10;
    % thresholdResObj = 1e-6;

    % update Pi
    lam1 = 0.1 % for col-sum and row-sum constraints, both gradients are same formula
    lam2 = 10 % for orthogonalty

    % Ax-Pi*Ay hyper-parameters
    lambda1 = 0.01;
    lambda2 = 0.01;
    lambda3 = 0.05;
    x0_init = 0.05;

    %% Initialize the CMTF rank R of tensor X and Y
    % trueRank = length(lambdas_data{1}) % hyper-parameter, need to guess rank of the tensor????
    cmtfRank = guessRank % trueRank
    fprintf(1, 'CMTF rank =%d \n', guessRank);
    initRank = guessRank; % the value of initRank starting from mode1_size
    fprintf(1, 'Initial rank =%d \n', initRank);


    %% Initial permutation matrix Pi
    % initPermu = eye(mode1_size,mode1_size);
    % initPermu = initPermu(randperm(mode1_size),:); % this is the init permutation matrix \Pi
    % initPermu = rand(mode1_size,mode1_size);
    % Initialize by smart initialization
    % [initPermu] = Initialization_1(X, Y);
    % [initPermu] = Initialization_2(X, Y, initRank); % best initialization of \Pi
    % [initPermu] = Initialization_3(X, Y);

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
    % initPermu = truePermu % test if Factor Matching Score is correct, FMS is too low for trueP=initP
    initial_permutation = initPermu; % store the initial permutation matrix 


    %% Initialize A, B, C, D, E by CPD or Random
    if strcmp(str_factor, 'CPD') % doing cpd initialization
        % Initialize A, B, C, D, E by CPD
        [Fac] = cp_als(X, cmtfRank,'tol',10^-9,'maxiters',1e3);
        A = Fac.U{1};
        B = Fac.U{2};
        C = Fac.U{3};
        Y_1modeproduct = ttm(Y, initPermu, 1);
        [Fac2] = cp_als(Y_1modeproduct, cmtfRank,'tol',10^-9,'maxiters',1e3);
        % Ay = Fac2.U{1}; % try CDP of Y*Pi
        % A = Fac2.U{1}; % try CDP of Y*Pi
        D = Fac2.U{2};
        E = Fac2.U{3};
    elseif strcmp(str_factor, 'rand') % doing cpd initialization
        % Initialize A, B, C, D, E by random
        A = rand(sz(1), guessRank);
        B = rand(sz(2), guessRank);
        C = rand(sz(3), guessRank);
        D = rand(sz(5), guessRank);
        E = rand(sz(6), guessRank);
    end



    %% matricization: unfolded tensor X and Y
    % One-mode matricization of Y
    Y1 = tenmat(Y, 1);
    Y1 = double(Y1);
    % Two-mode matricization of Y
    Y2 = tenmat(Y, 2);
    Y2 = double(Y2);
    % Thrid-mode matricization of Y
    Y3 = tenmat(Y, 3);
    Y3 = double(Y3);

    %% function values
    func_val(1) = 0;
    Fac_val(1) = 0;
    Pi_val(1) = 0;
    accuracy_pi(1) = 0;


    fprintf(1, 'Run objctive function 1 algorithm...\n');
    iter = 1;

    while(iter <= MaxMostOuterIter)
        %% 6. update Pi
        EkrD = khatrirao(E, D);
        Y1ktrED = Y1 * EkrD; % equaivlent to Y1*khatrirao(E, D)
        Y1ktrED_times_AT = (Y1ktrED * (A.')).';
        Y1Y1T = Y1 * (Y1.');
        
        [alignment_matrix] = LS_solver_f1_conTest(Y1Y1T, Y1ktrED_times_AT, lambda1, lambda2, lambda3, x0_init, mode1_size, str_row, str_col, str_entropy); %'noRow', 'noCol', 'noEntropy');
        alignment_matrix
        imagesc(alignment_matrix);

        % alignment_matrix = truePermu;
        initPermu = alignment_matrix;
        % alignment_matrix = initPermu;


        % calculate the mode-1 product of Y times initPi
        Y_1modeproduct = ttm(Y, initPermu, 1);
        
        %% 1. update A
        % Y1_ = tenmat(Y_1modeproduct, 1); % mode-1 matricization of tensor (Y *1 Pi)
        % Y1_ = double(Y1_);
        % EkrD = khatrirao(E, D);
        % Y1_ktrED = Y1_ * EkrD; % equaivlent to Y1_*khatrirao(E, D)
        
        Uy = {A, D, E};
        Y1_ktrED = mttkrp(Y_1modeproduct, Uy, 1);

        Ux = {A, B, C};
        X1ktrCB = mttkrp(X, Ux, 1); % equaivlent to X1*khatrirao(C, B)
        CTC_times_BTB = ((C.')*C) .* ((B.')*B);
        ETE_times_DTD = ((E.')*E) .* ((D.')*D);
        [szCTC_ETE, ~] = size(ETE_times_DTD);

        A = (X1ktrCB + Y1_ktrED) / (CTC_times_BTB + ETE_times_DTD + 10^(-7)*eye(szCTC_ETE));
        % A = trueA;

        %% 2. update B
        Ux = {A, B, C};
        X2ktrCA = mttkrp(X, Ux, 2); % equaivlent to X2*khatrirao(C, A)
        CTC_times_ATA = ((C.')*C) .* ((A.')*A);
        [szCTC_ATA, ~] = size(CTC_times_ATA);
        
        B = X2ktrCA / (CTC_times_ATA + 10^(-7)*eye(szCTC_ATA));
        % B = trueB;

        %% 3. update C
        Ux = {A, B, C};
        X3ktrBA = mttkrp(X, Ux, 3); % equaivlent to X3*khatrirao(B, A)
        BTB_times_ATA = ((B.')*B) .* ((A.')*A);
        [szBTB_ATA, ~] = size(BTB_times_ATA);

        C = X3ktrBA / (BTB_times_ATA + 10^(-7)*eye(szBTB_ATA));
        % C = trueC;

        %% 4. update D
        % Y2_ = tenmat(Y_1modeproduct, 2); % mode-2 matricization of tensor (Y *1 Pi)
        % Y2_ = double(Y2_);
        % EkrA = khatrirao(E, A);
        % Y2_ktrEA = Y2_ * EkrA;

        Uy = {A, D, E};
        Y2_ktrEA = mttkrp(Y_1modeproduct, Uy, 2);
        ETE_times_ATA = ((E.')*E) .* ((A.')*A);
        [szETE_ATA, ~] = size(ETE_times_ATA);

        D = Y2_ktrEA / (ETE_times_ATA + 10^(-7)*eye(szETE_ATA));
        % D = trueD;

        %% 5. update E
        % Y3_ = tenmat(Y_1modeproduct, 3); % mode-3 matricization of tensor (Y *1 Pi)
        % Y3_ = double(Y3_);
        % DkrA = khatrirao(D, A);
        % Y3_ktrDA = Y3_ * DkrA;

        Uy = {A, D, E};
        Y3_ktrDA = mttkrp(Y_1modeproduct, Uy, 3);
        DTD_times_ATA = ((D.')*D) .* ((A.')*A);
        [szDTD_ATA, ~] = size(DTD_times_ATA);

        E = Y3_ktrDA / (DTD_times_ATA + 10^(-7)*eye(szDTD_ATA));
        % E = trueE;

        %% 6. update Pi
        % EkrD = khatrirao(E, D);
        % Y1ktrED = Y1 * EkrD; % equaivlent to Y1*khatrirao(E, D)
        % Y1ktrED_times_AT = (Y1ktrED * (A.')).';
        % Y1Y1T = Y1 * (Y1.');
        
        % [alignment_matrix] = LS_solver_Pi(Y1Y1T, Y1ktrED_times_AT, lambda1, lambda2, lambda3, x0_init, mode1_size);
        % alignment_matrix
        % imagesc(alignment_matrix);

        % % alignment_matrix = truePermu;
        % initPermu = alignment_matrix;

        %% 7. calculate Ax and Ay
        Ay = pinv(initPermu) * A;
        Ax = A;


        %% 9. update loss here
        %% evaluate the obj function: min ||X - [A, B, C]||_fro^2 + ||Y*Pi - [A, D, E]||_fro^2
        X_new = ktensor(ones(1,guessRank)',{A, B, C});
        X_new = tensor(X_new);
        f_X = norm(X - X_new)^2;
        Y_1modeproduct = ttm(Y, alignment_matrix, 1);
        Y_new = ktensor(ones(1,guessRank)',{A, D, E});
        Y_new = tensor(Y_new);
        f_Y = norm(Y_1modeproduct - Y_new)^2;
        f_objloss = f_X + f_Y;
        %% Calculate the loss of ||Ax - \Pi*Ay||fro
        f_factors = norm(Ax - alignment_matrix * Ay, "fro")^2; 
        %% Calculate the loss of alignment_matrix and the true permutation matrix
        f_pi = norm(alignment_matrix - truePermu, "fro")^2;

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
    Res = cell(6,1);
    Res{1} = func_val;
    Res{2} = Fac_val;
    Res{3} = Pi_val;
    Res{4} = accuracy_pi;

    % final Loss
    Loss_obj = f_objloss;
    Loss_pi = f_pi;

    % calculating the accuracy of the learned Pi
    ACC_Pi = acc_pi;
    % kmeans_acc_pi = 0;
    % for i = 1:mode1_size
    %     for j = 1:mode1_size
    %         if truePermu(i,j)==1 && alignment_matrix(i,j)>=0.4
    %             kmeans_acc_pi = kmeans_acc_pi+1;
    %         elseif truePermu(i,j)==0 %
    %             % if row i and j are in same cluster: Cidx(i)==Cidx(j), then count it as true
    %             % otherwise row i and j are not in same cluster, j cannot change to i-th position!!
    %             if alignment_matrix(i,j)>=0.4 && Cidx(i)==Cidx(j)
    %                 kmeans_acc_pi = kmeans_acc_pi+1;
    %             end
    %         end
    %     end
    % end
    % kmeans_acc_pi = kmeans_acc_pi/mode1_size;
    % ACC_Pi = kmeans_acc_pi

end