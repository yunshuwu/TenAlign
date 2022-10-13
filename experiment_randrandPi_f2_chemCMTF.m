%%  test script for initialization methods for factors

close all
clear all
%% add AO-ADMM solver functions to path
addpath(genpath('functions'))
%% add REGAL alignment solver to path
addpath(genpath('alignments'))
%% add other apckages to your path!
% addpath(genpath('...\tensor_toolbox-v3.1')) %Tensor toolbox is needed!  MATLAB Tensor Toolbox. Copyright 2017, Sandia Corporation, http://www.tensortoolbox.org/
% addpath(genpath('...\L-BFGS-B-C-master')) % LBFGS-B implementation only needed when other loss than Frobenius is used, download here: https://github.com/stephenbeckr/L-BFGS-B-C
% addpath(genpath('...\proximal_operators\code\matlab')) % Proximal operator repository needed! download here: http://proximity-operator.net/proximityoperator.html
% addpath(genpath('../deltacon')) % DeltaCon similarity metric needed!
addpath(genpath('../CMTF_Toolbox_v1_1')) % CMTF toolbox is needed!
addpath(genpath('../poblano_toolbox-main')) % poblano_toolbox-main is needed!
addpath(genpath('../tensor_toolbox-v3.1')) %Tensor toolbox is needed!  MATLAB Tensor Toolbox. Copyright 2017, Sandia Corporation, http://www.tensortoolbox.org/
addpath(genpath('../L-BFGS-B-C-master')) % LBFGS-B implementation only needed when other loss than Frobenius is used, download here: https://github.com/stephenbeckr/L-BFGS-B-C
addpath(genpath('../proximal_operators/code/matlab')) % Proximal operator repository needed! download here: http://proximity-operator.net/proximityoperator.html
addpath(genpath('../nway331'))
addpath(genpath('functions')) 
addpath(genpath('alignments'))

% load CMTF data
load('datasets/EEM_NMR_LCMS.mat')
Xnew = tensor(X.data); % 28*251*21 tensor
Ynew = tensor(Y.data); % 28*13324*8 tensor
Znew = double(Z.data); % 28*168 matrix, make this a 3-mode tensor to test our algorithm!

% turn all tensons into multidim-arrays
Xnew = double(Xnew);
Ynew = double(Ynew);
Znew = double(Znew);
%% size of input tensors
szX = size(Xnew);
szY = size(Ynew);
szZ = size(Znew);

%% Before normalize the whole tensor, first do colunm normalization for Z, for X and Y do "column" normalization
for c = 1:szZ(2)
    % normalize all columns of Z
    Znew(:,c) = Znew(:,c)/norm(Znew(:,c));
end
for k = 1:szX(3) % K = szX(3)
    % normalize all col of X and Y
    for c = 1:szX(2) % J = szX(2)
        % X(:,:,k)(:,c) = X(:,:,k)(:,c)/norm(X(:,:,k)(:,c));
        Xnew(:,c,k) = Xnew(:,c,k)/norm(Xnew(:,c,k));
    end
end
for k = 1:szY(3) % K = szY(3)
    % normalize all col of and Y
    for c = 1:szY(2) % J = szY(2)
        % Y(:,:,k)(:,c) = Y(:,:,k)(:,c)/norm(Y(:,:,k)(:,c));
        Ynew(:,c,k) = Ynew(:,c,k)/norm(Ynew(:,c,k));
    end
end

%% applying centering and scaling to chem-tensors
% Cent = [1 1 1]; % centering across all modes
% Scal = [1 1 1]; % same scaling across all modes
% % apply N-way preprocessing to chemistry tensors in multi-array format
% [Xnew,XMeans,XScales]=nprocess_0(Xnew,Cent,Scal);
% [Ynew,YMeans,YScales]=nprocess_0(Ynew,Cent,Scal);
% [Znew,ZMeans,ZScales]=nprocess_0(Znew,Cent,Scal);

% test if centering and scaling helps
% Xnew = normalize(Xnew);
% Ynew = normalize(Ynew);
% Znew = normalize(Znew);

%% copy matrix Z to a tensor with thrid-mode==1
Ztemp = tenones([szZ(1) szZ(2) 1]);
Ztemp(:,:,1) = Znew;
fprintf(1, 'create a one tensor...\n');
szZ = size(Ztemp);

sz = [szY szZ]

%% change multidim array Xnew,Ynew,Znew back to tensors
Xnew = tensor(Xnew);
Ynew = tensor(Ynew);
Ztemp = tensor(Ztemp);

%% After normalize columns, normalize whole tensor (follow what AO-ADMM does, devide tensors by their norms)
Xnew = Xnew/norm(Xnew);
Ynew = Ynew/norm(Ynew);
Ztemp = Ztemp/norm(Ztemp);

% input data tensors
Z.object{1} = Ynew;
Z.object{2} = Ztemp;

% create true permutation matrix
truePermu = eye(sz(1), sz(1));
truePermu = truePermu(randperm(sz(1)),:); % this is the true permutation matrix \Pi

% mode1_size = sz(1);
number_testcase = 100;
MaxMostOuterIter = 200; % change this "budget"!!
thresholdResObj = 1e-6;
guessRank = 4;
k_cluster = 4;
lambda = 10;

fprintf(1, 'Testing chemistry dataset...\n');

fms_factors(1) = 0;
fms_pi(1) = 0;
loss_obj(1) = 0;
loss_pi(1) = 0;
AccPi(1) = 0;

% load('fms_factors_CPDrandPi_Max200_f2.mat', 'fms_factors')
% load('fms_pi_CPDrandPi_Max200_f2.mat', 'fms_pi')
load('loss_obj_CPDrandPi_Max200_f2.mat', 'loss_obj')
load('loss_pi_CPDrandPi_Max200_f2.mat', 'loss_pi')
load('AccPi_CPDrandPi_Max200_f2.mat', 'AccPi')
[~,num_round] =  size(fms_factors)

iter = num_round + 1;
% iter=1;
while(iter <= number_testcase)
    % [learnedPermu, initPermu, accPi, Loss_obj, Loss_pi] = tensor_align_f1_chemCMTF(Z, MaxMostOuterIter, thresholdResObj, 'CPD', 'randPi', k_cluster, guessRank, truePermu, '', '', '');

    [learnedPermu, initPermu, accPi, Loss_obj, Loss_pi] = tensor_align_f2_chemCMTF(Z, MaxMostOuterIter, thresholdResObj, lambda, 'CPD', 'randPi', k_cluster, guessRank, truePermu, '', '', '');
   
    % [learnedPermu, initPermu, accPi, Loss_obj, Loss_pi] = tensor_align_f2_REGAL_realTensor(Z, MaxMostOuterIter, thresholdResObj, lambda, 'rand', 'randPi', k_cluster, guessRank, truePermu, '', '', '');
    
    % [alignment_matrix, initial_permutation, ACC_Pi, Loss_obj, Loss_pi] = tensor_align_f2_REGAL_realTensor(Z, MaxMostOuterIter, thresholdResObj, lambda, str_factor, str_pi, k_cluster, guessRank, truePermu, str_row, str_col, str_entropy);
    % [alignment_matrix, initial_permutation, ACC_Pi, Loss_obj, Loss_pi] = tensor_align_f2_chemCMTF(Z, MaxMostOuterIter, thresholdResObj, str_factor, str_pi, k_cluster, guessRank, truePermu, str_row, str_col, str_entropy)
    % [alignment_matrix, initial_permutation, ACC_Pi, Loss_obj, Loss_pi] = tensor_align_f1_chemCMTF(Z, MaxMostOuterIter, thresholdResObj, str_factor, str_pi, k_cluster, guessRank, truePermu);


    % save these two values every iteration
    % fms_factors(iter) = FMS_Factor;
    % fms_pi(iter) = FMS_Pi;
    loss_obj(iter) = Loss_obj;
    loss_pi(iter) = Loss_pi;
    AccPi(iter) = accPi;

    % save('fms_factors_CPDrandPi_Max200_f2.mat','fms_factors')
    % save('fms_pi_CPDrandPi_Max200_f2.mat','fms_pi')
    save('loss_obj_CPDrandPi_Max200_f2.mat','loss_obj')
    save('loss_pi_CPDrandPi_Max200_f2.mat','loss_pi')
    save('AccPi_CPDrandPi_Max200_f2.mat','AccPi')

    iter = iter+1;
end

% Edges = ones(100 ,1);
% Edges = Edges * 0.01;
% disp(Edges)
% Edges = cumsum(Edges)

% figure()
% subplot(5,1,1)
% h1 = histogram(fms_factors,Edges)
% % h1.Normalization = 'probability';
% xlabel('FMS Factor')
% ylabel('num')

% subplot(5,1,2)
% h2 = histogram(fms_pi,Edges)
% % h2.Normalization = 'probability';
% xlabel('FMS Pi value')
% ylabel('num')

% subplot(5,1,3)
% h3 = histogram(loss_obj,Edges)
% % h3.Normalization = 'probability';
% xlabel('obj loss value')
% ylabel('num')

% subplot(5,1,4)
% h4 = histogram(loss_pi)
% %h4.BinWidth = 50;
% % h4.Normalization = 'probability';
% xlabel('Pi loss value')
% ylabel('num')

% subplot(5,1,5)
% h5 = histogram(AccPi,Edges)
% % h5.Normalization = 'probability';
% xlabel('Pi accuracy value')
% ylabel('num')

% saveas(gcf,'figure_CPDrandPi_Max200.png')

% % LearnedPi = 'learnedPi.png';
% % imwrite( ind2rgb(im2uint8(mat2gray(learnedPermu)), parula(256)), LearnedPi)
% % truePi = 'truePi.png';
% % imwrite( ind2rgb(im2uint8(mat2gray(truePermu)), parula(256)), truePi)