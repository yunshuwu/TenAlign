function [alignment_matrix, truePermu, initial_permutation, ACC_Pi, FMS_Factor, FMS_Pi, Loss_obj, Loss_pi] = tensor_align_f2(mode1_size, MaxMostOuterIter, thresholdResObj, lambda, str_factor, str_pi, k_cluster)
    % alignment_matrix: learned permutation matrix
    % truePermu: true permutation matrix
    % initial_permutation: the initialization pi matrix

%% specify synthetic data
% mode1_size = 10;
sz     = [mode1_size 30 40 mode1_size 70 10]; %size of each mode
P      = 2; %number of tensors
lambdas_data= {[1 1 1 1 1], [1 1 1 1 1]}; % norms of components in each data set (length of each array specifies the number of components in each dataset)
modes  = {[1 2 3], [4 5 6]}; % which modes belong to which dataset: every mode should have its unique number d, sz(d) corresponds to size of that mode
noise = 0; %level of noise, for gaussian noise only!
distr_data = {@(x,y) rand(x,y),@(x,y) randn(x,y),@(x,y) randn(x,y),@(x,y) rand(x,y),@(x,y) rand(x,y),@(x,y) randn(x,y)}; % function handle of distribution of data within each factor matrix /or Delta if linearly coupled, x,y are the size inputs %coupled modes need to have same distribution! If not, just the first one will be considered
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
init_options.lambdas_init = {[1 1 1 1 1], [1 1 1 1 1]}; %norms of components in each data set for initialization
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
    Z.object{p} = X{p}; % Z.object{} is also ktensor!
    normZ{p} = norm(Z.object{p}); % norm(X) is the largest singular value of X
    Z.object{p} = Z.object{p}/normZ{p};
end

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

% Ay hyper-parameters
MaxAyIter = 10000;
% lambda = 50%800 % control how close two spaces are
alpha_y = 0.00001; % for calculating Ay by gradient descent
thresholdResidualAy = 1e-6; % residuals of updating Ay

% Ax-Pi*Ay hyper-parameters
lambda1 = 0.01;
lambda2 = 0.01;
lambda3 = 0.1;
x0_init = 0.05;

%% Initialize the CMTF rank R of tensor X and Y
trueRank = length(lambdas_data{1}) % hyper-parameter, need to guess rank of the tensor????
cmtfRank = trueRank
fprintf(1, 'CMTF rank =%d \n', trueRank);
initRank = 10; % the value of initRank starting from mode1_size
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
    [Fac] = cp_als(X, trueRank,'tol',10^-9,'maxiters',1e3);
    Ax = Fac.U{1};
    B = Fac.U{2};
    C = Fac.U{3};
    Y_1modeproduct = ttm(Y, initPermu, 1);
    [Fac2] = cp_als(Y_1modeproduct, trueRank,'tol',10^-9,'maxiters',1e3);
    Ay = Fac2.U{1}; % try CDP of Y*Pi
    % A = Fac2.U{1}; % try CDP of Y*Pi
    D = Fac2.U{2};
    E = Fac2.U{3};
elseif strcmp(str_factor, 'rand') % doing cpd initialization
    % Initialize Ax, B, C, Ay, D, E by random
    Ax = rand(sz(1), trueRank);
    B = rand(sz(2), trueRank);
    C = rand(sz(3), trueRank);
    Ay = rand(sz(4), trueRank);
    D = rand(sz(5), trueRank);
    E = rand(sz(6), trueRank);
end

%% function values
func_val(1) = 0;
Fac_val(1) = 0;
Pi_val(1) = 0;
func_match_score_xy(1) = 0;
func_match_score_xypi(1) = 0;
accuracy_pi(1) = 0;

    %% --------------  Here should be in the MostOuterIter  -------------- %
fprintf(1, 'Run our algorithm...\n');
iter = 1;

while(iter <= MaxMostOuterIter)
    % fprintf(1, 'No.%d iteration: \n', iter);
    %% 1. update Ax: 
    Ux = {Ax, B, C};
    X1ktrCB = mttkrp(X, Ux, 1); % equaivlent to X1*khatrirao(C, B)
    % optimization: X1*(C kr B) is too expensive, find matlab implementation of MTTKRP (matrixize tensor times kr product)
    CTC_times_BTB = ((C.')*C) .* ((B.')*B);
    [sizeCTCBTB, ~] = size(CTC_times_BTB);
    IdCB = eye(sizeCTCBTB, sizeCTCBTB);
    CB = CTC_times_BTB + lambda*IdCB; % mrdivide, /; x = B/A
    Ax = (X1ktrCB + lambda * initPermu * Ay) / (CB + 10^(-7)*eye(sizeCTCBTB));

    %% 2. update B
    Ux = {Ax, B, C};
    X2ktrCAx = mttkrp(X, Ux, 2); % equaivlent to X2*khatrirao(C, Ax)
    CTC_times_AxTAx = ((C.')*C) .* ((Ax.')*Ax);
    CAx = CTC_times_AxTAx;
    [sizeCTCAxTAx, ~] = size(CTC_times_AxTAx);
    B = X2ktrCAx / (CAx + 10^(-7)*eye(sizeCTCAxTAx));

    %% 3. update C
    Ux = {Ax, B, C};
    X3ktrBAx = mttkrp(X, Ux, 3); % equaivlent to X3*khatrirao(B, Ax)
    BTB_times_AxTAx = ((B.')*B) .* ((Ax.')*Ax);
    BAx = BTB_times_AxTAx;
    [sizeBTBAxTAx, ~] = size(BTB_times_AxTAx);
    C = X3ktrBAx / (BAx + 10^(-7)*eye(sizeBTBAxTAx));

    %% 4. update Ay by Gradient Descent
    Uy = {Ay, D, E};
    Y1ktrED = mttkrp(Y, Uy, 1); % equaivlent to Y1*khatrirao(E, D)
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
    Y2ktrEAy = mttkrp(Y, Uy, 2); % equaivlent to Y2*khatrirao(E, Ay)
    ETE_times_AyTAy = ((E.')*E) .* ((Ay.')*Ay);
    EAy = ETE_times_AyTAy;
    [sizeETEAyTAy, ~] = size(ETE_times_AyTAy);
    D = Y2ktrEAy / (EAy + 10^(-5)*eye(sizeETEAyTAy));

    %% 6. update E
    Uy = {Ay, D, E};
    Y3ktrDAy = mttkrp(Y, Uy, 3); % equaivlent to Y3*khatrirao(D, Ay)
    % ktrDAy = khatrirao(D, Ay);
    DTD_times_AyTAy = ((D.')*D) .* ((Ay.')*Ay);
    DAy = DTD_times_AyTAy;
    [sizeDTDAyTAy, ~] = size(DTD_times_AyTAy);
    E = Y3ktrDAy / (DAy + 10^(-5)*eye(sizeDTDAyTAy));

    % %% Update the column ordering of Ay D E
    % FacAx = ktensor(ones(length(lambdas_data{1}),1), Ax);
    % FacAx = normalize(FacAx);
    % FacAy = ktensor(ones(length(lambdas_data{1}),1), Ay);
    % FacAy = normalize(FacAy);
    % % need PiColXY to do "most correct initialization"
    % [FMSAyAx,~,~,PiColXY] = score(FacAy, FacAx);
    % XYPermu = eye(trueRank, trueRank);
    % XYPermu = XYPermu(:,PiColXY)
    % % Ay * PiColXY = Ax
    % % Update Ay, D, E by PiColXY
    % Ay = Ay * XYPermu;
    % D = D * XYPermu;
    % E = E * XYPermu;

    % % Check if Ax and Ay has same column order
    % FacAx = ktensor(ones(length(lambdas_data{1}),1), Ax);
    % FacAx = normalize(FacAx)
    % FacAy = ktensor(ones(length(lambdas_data{1}),1), Ay);
    % FacAy = normalize(FacAy)
    % [~,~,~,PiColXY] = score(FacAy, FacAx)


    %% 7. update Pi by solver
    %% Learn permutation matrix by solving func with non-linear constraints
    [alignment_matrix] = LS_nonlcon_solver(Ax, Ay, lambda1, lambda2, lambda3, x0_init);
    % normalize initPermu to sumRow(initPermu)=1
    % sumRowPi = sum(alignment_matrix, 2); % sum of each row
    % sumRowPi = sumRowPi.';
    % alignment_matrix = bsxfun(@rdivide, alignment_matrix, sumRowPi(:)); % divide each row of the matrix with corresponding element in row vector sumRow
    
    % Update initial permutation matrix: initPermu
    % imagesc(alignment_matrix);
    initPermu = alignment_matrix;
    alignment_matrix

    %% 8 Calculate FMS regarding Ax and Ay
    FacNorm = cell(P,1);
    FacNorm{1} = ktensor(ones(length(lambdas_data{1}),1),Ax,B,C);
    FacNorm{2} = ktensor(ones(length(lambdas_data{2}),1),Ay,D,E);
    FacNorm{1} = normalize(FacNorm{1});
    FacNorm{2} = normalize(FacNorm{2});
    % FMS_xy
    % fprintf(1, 'true factors 1.. \n');
    % true_ktensor{1}
    % fprintf(1, 'true factors 2.. \n');
    % true_ktensor{2}
    [FMS1_xy, ~,~, PiColAxBC] = score(FacNorm{1},true_ktensor{1});
    [FMS2_xy, ~,~, PiColAyDE] = score(FacNorm{2},true_ktensor{2});
    PiColAxBC
    PiColAyDE
    FMStotal_xy = FMS1_xy*FMS2_xy;
    % fprintf(1, 'factor matching score...');

    %% Also measure FMS(Ax, Pi*Ay)
    M1 = ktensor(ones(length(lambdas_data{1}),1), Ax);
    M1 = normalize(M1);
    M2 = ktensor(ones(length(lambdas_data{1}),1), initPermu*Ay);
    M2 = normalize(M2);
    % [FMStotal_xypi,~,~,PiCol] = score(Ax, initPermu*Ay);
    [FMStotal_xypi,~,~,PiCol] = score(M1, M2);
    PiCol

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
        fprintf(1,' MostOuterIter  f loss func    f factors      f permutation       FMS xy         FMS xypi      Pi Accuracy  \n');
        fprintf(1,'--------------  -------------  -------------  ---------------  -------------  --------------  --------------\n');
    end
    fprintf(1,'%7d %16f %16f %16f %16f %16f %16f\n', iter, f_objloss, f_factors, f_pi, FMStotal_xy, FMStotal_xypi, acc_pi);

    func_val(iter) = f_objloss;
    Fac_val(iter) = f_factors;
    Pi_val(iter) = f_pi;
    func_match_score_xy(iter) = FMStotal_xy;
    func_match_score_xypi(iter) = FMStotal_xypi;
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
Res{4} = func_match_score_xy;
Res{5} = func_match_score_xypi;
Res{6} = accuracy_pi;

% final Loss
Loss_obj = f_objloss;
Loss_pi = f_pi;

% usually high FMS_factors score means better permutation matrix
FMS_Factor = FMStotal_xy; 
FMS_Pi = FMStotal_xypi;

% calculating the accuracy of the learned Pi
ACC_Pi = acc_pi
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