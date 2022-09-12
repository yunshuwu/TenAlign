function [initPermu] = Initialization_1(X, Y_org)

%% Initlization permutation matrix \Pi
%% Creating the Ax_init and Ay_init
Ax_init = X(:, :, 1); % take the first slice of tensor X
Ax_init = tenmat(Ax_init, 1);
Ax_init = double(Ax_init);
% [rAx, cAx] = size(Ax_init);
% fprintf(1, 'size of Ax slice = [%d,%d] \n', rAx, cAx);
Ay_init = Y_org(:, :, 1); % take the first slice of tensor Y_org (the one that's not multiply by \Pi)
Ay_init = tenmat(Ay_init, 1);
Ay_init = double(Ay_init);
% [rAy, cAy] = size(Ay_init);
% fprintf(1, 'size of Ay slice = [%d,%d] \n', rAy, cAy);

[Ux, Sx, Vx] = svd(Ax_init);
[row_x, col_x] = size(Ux);
% fprintf(1, 'size of Ux factor matrix = [%d,%d] \n', row_x, col_x);
[Uy, Sy, Vy] = svd(Ay_init);
% [row_y, col_y] = size(Uy);
% fprintf(1, 'size of Uy factor matrix = [%d,%d] \n', row_y, col_y);

Ax_init = Ux(:, 1:row_x);
Ay_init = Uy(:, 1:row_x);
% [rAy, cAy] = size(Ay_init);
% fprintf(1, 'size of Ay slice = [%d,%d] \n', rAy, cAy);

%% Learn the initial permutation by Ax_init and Ay_init
AyinitT = Ay_init.'; % transpose Uy for Ay_init
[N_init, ~] = size(Ax_init);
In_init = eye(N_init, N_init);
C = kron(AyinitT, In_init);
d = Ax_init(:); % vec(Ax_init) = ((R*N_init) * 1)
lb = zeros(N_init*N_init, 1);
ub = ones(N_init*N_init, 1);
% calculate Aeq and beq
Aeq = zeros(N_init, N_init*N_init);
beq = ones(N_init, 1);
for i = 1:N_init
    for j = 0:(N_init-1)
        Aeq(i, j*N_init+i) = 1;
    end
end
%% calculate vec(Pi) by lsqlin solver
initPi = lsqlin(C, d, [], [], Aeq, beq, lb, ub);
fprintf(1, 'initial vector Pi: \n');
size(initPi)
initPermu = reshape(initPi, N_init, N_init);
% %% normalize initPermu to sumRow(initPermu)=1
% sumRow = sum(initPermu, 2); % sum of each row
% sumRow = sumRow.';
% initPermu = bsxfun(@rdivide, initPermu, sumRow(:)); % divide each row of the matrix with corresponding element in row vector sumRow

% initial_permutation = initPermu;

imagesc(initPermu);
fprintf(1, 'smart initial version 1 \n');
initPermu

end