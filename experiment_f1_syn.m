%%  test script for initialization methods for factors
close all
clear all
%% add AO-ADMM solver functions to path
addpath(genpath('functions'))
%% add REGAL alignment solver to path
addpath(genpath('alignments'))
%% add other apckages to your path!
addpath(genpath('../CMTF_Toolbox_v1_1')) % CMTF toolbox is needed!
addpath(genpath('../poblano_toolbox-main')) % poblano_toolbox-main is needed!
addpath(genpath('../tensor_toolbox-v3.1')) %Tensor toolbox is needed!  MATLAB Tensor Toolbox. Copyright 2017, Sandia Corporation, http://www.tensortoolbox.org/
addpath(genpath('../L-BFGS-B-C-master')) % LBFGS-B implementation only needed when other loss than Frobenius is used, download here: https://github.com/stephenbeckr/L-BFGS-B-C
addpath(genpath('../proximal_operators/code/matlab')) % Proximal operator repository needed! download here: http://proximity-operator.net/proximityoperator.html
addpath(genpath('functions')) 
addpath(genpath('alignments'))

mode1_size = 10;
number_testcase = 10;
MaxMostOuterIter = 200; % change this "budget"!!
thresholdResObj = 1e-9;
k_cluster = mode1_size; % means no cluster allowed
trueRank = 4%10;
guessRank = 4%10;

fprintf(1, 'Testing rank value for formula 1...\n');

loss_obj(1) = 0;
loss_pi(1) = 0;
AccPi(1) = 0;

iter=1;
while(iter <= number_testcase)
    [learnedPermu, truePermu, initPermu, accPi, Loss_obj, Loss_pi] = tensor_align_f1_conTest(mode1_size, MaxMostOuterIter, thresholdResObj, 'rand', 'randPi', k_cluster, trueRank, guessRank, ' ', ' ', ' ');

    % save these two values every iteration
    loss_obj(iter) = Loss_obj;
    loss_pi(iter) = Loss_pi;
    AccPi(iter) = accPi;

    save('loss_obj_uncon_f1.mat','loss_obj')
    save('loss_pi_uncon_f1.mat','loss_pi')
    save('AccPi_uncon_f1.mat','AccPi')

    iter = iter+1;
end

Edges = ones(100 ,1);
Edges = Edges * 0.01;
disp(Edges)
Edges = cumsum(Edges)

subplot(3,1,1)
h3 = histogram(loss_obj,Edges)
xlabel('obj loss value')
ylabel('num')

subplot(3,1,2)
h4 = histogram(loss_pi)
xlabel('Pi loss value')
ylabel('num')

subplot(3,1,3)
h5 = histogram(AccPi,Edges)
xlabel('Pi accuracy value')
ylabel('num')

saveas(gcf,'figure_uncon_f1.png')