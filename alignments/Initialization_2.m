function [initPermu] = Initialization_2(X, Y_org, initRank)

    %% Initlization permutation matrix \Pi
    %% Creating the Ax_init and Ay_init
    Xm = tenmat(X, 1); %<-- Same as Ax = tenmat(X,1,2:3)
    Xm = double(Xm);
    [rAx, cAx] = size(Xm);
    fprintf(1, 'size of Xm slice = [%d,%d] \n', rAx, cAx);
    
    Ym = tenmat(Y_org, 1);
    Ym = double(Ym);
    % [rAy, cAy] = size(Ay_init);
    % fprintf(1, 'size of Ay slice = [%d,%d] \n', rAy, cAy);
    
    [Ux, Sx, Vx] = svd(Xm);
    [row_x, col_x] = size(Ux);
    fprintf(1, 'size of Ux factor matrix = [%d,%d] \n', row_x, col_x);
    [Uy, Sy, Vy] = svd(Ym);
    % [row_y, col_y] = size(Uy);
    % fprintf(1, 'size of Uy factor matrix = [%d,%d] \n', row_y, col_y);
    
    Ax_init = Ux(:, 1:initRank);
    Ay_init = Uy(:, 1:initRank);
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
    options = optimoptions('lsqlin','Algorithm','interior-point','Display','off','MaxIter',1000);
    initPi = lsqlin(C, d, Aeq, beq, [], [], lb, ub, [], options);
    
    fprintf(1, 'initial vector Pi: \n');
    size(initPi)
    initPermu = reshape(initPi, N_init, N_init);
    % %% normalize initPermu to sumRow(initPermu)=1
    % sumRow = sum(initPermu, 2); % sum of each row
    % sumRow = sumRow.';
    % initPermu = bsxfun(@rdivide, initPermu, sumRow(:)); % divide each row of the matrix with corresponding element in row vector sumRow
    
    % initial_permutation = initPermu;
    
    imagesc(initPermu);
    fprintf(1, 'smart initial version 2 \n');
    initPermu
    
end