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
addpath(genpath('functions')) 
addpath(genpath('alignments'))

mode1_size = 15;
number_testcase = 100;
MaxMostOuterIter = 200; % change this "budget"!!
thresholdResObj = 1e-7;
k_cluster = mode1_size;
lambda = 5;

fprintf(1, 'Testing formula 2 with different lambda...\n');

fms_factors(1) = 0;
fms_pi(1) = 0;
loss_obj(1) = 0;
loss_pi(1) = 0;
AccPi(1) = 0;

% load('fms_factors_randrandPi_lam5_f2.mat', 'fms_factors')
% load('fms_pi_randrandPi_lam5_f2.mat', 'fms_pi')
% load('loss_obj_randrandPi_lam5_f2.mat', 'loss_obj')
% load('loss_pi_randrandPi_lam5_f2.mat', 'loss_pi')
% load('AccPi_randrandPi_lam5_f2.mat', 'AccPi')
% [~,num_round] =  size(fms_factors)

% iter = num_round + 1;
iter=1;
while(iter <= number_testcase)
    [learnedPermu, truePermu, initPermu, accPi, FMS_Factor, FMS_Pi, Loss_obj, Loss_pi] = tensor_align_f2(mode1_size, MaxMostOuterIter, thresholdResObj, lambda, 'rand', 'randPi', k_cluster);

    % save these two values every iteration
    fms_factors(iter) = FMS_Factor;
    fms_pi(iter) = FMS_Pi;
    loss_obj(iter) = Loss_obj;
    loss_pi(iter) = Loss_pi;
    AccPi(iter) = accPi;

    save('fms_factors_randrandPi_lam5_f2.mat','fms_factors')
    save('fms_pi_randrandPi_lam5_f2.mat','fms_pi')
    save('loss_obj_randrandPi_lam5_f2.mat','loss_obj')
    save('loss_pi_randrandPi_lam5_f2.mat','loss_pi')
    save('AccPi_randrandPi_lam5_f2.mat','AccPi')

    iter = iter+1;
end

Edges = ones(100 ,1);
Edges = Edges * 0.01;
disp(Edges)
Edges = cumsum(Edges)

figure()
subplot(5,1,1)
h1 = histogram(fms_factors,Edges)
% h1.Normalization = 'probability';
xlabel('FMS Factor')
ylabel('num')

subplot(5,1,2)
h2 = histogram(fms_pi,Edges)
% h2.Normalization = 'probability';
xlabel('FMS Pi value')
ylabel('num')

subplot(5,1,3)
h3 = histogram(loss_obj,Edges)
% h3.Normalization = 'probability';
xlabel('obj loss value')
ylabel('num')

subplot(5,1,4)
h4 = histogram(loss_pi)
%h4.BinWidth = 50;
% h4.Normalization = 'probability';
xlabel('Pi loss value')
ylabel('num')

subplot(5,1,5)
h5 = histogram(AccPi,Edges)
% h5.Normalization = 'probability';
xlabel('Pi accuracy value')
ylabel('num')

saveas(gcf,'figure_randrandPi_lam5_f2.png')

% LearnedPi = 'learnedPi.png';
% imwrite( ind2rgb(im2uint8(mat2gray(learnedPermu)), parula(256)), LearnedPi)
% truePi = 'truePi.png';
% imwrite( ind2rgb(im2uint8(mat2gray(truePermu)), parula(256)), truePi)