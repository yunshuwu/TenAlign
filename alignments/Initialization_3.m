function [initPermu] = Initialization_3(X, Y_org)%, mode1_size)

    %% Initlization permutation matrix \Pi
    % take num_pivots sampling slices
    % num_pivots = log2(mode1_size);

    %% Creating the Ax_init and Ay_init by sampling
    % for i = 1:num_pivots
    % end
    
    Ax1 = X(:, :, 1); % take the first slice of tensor X
    Ax1 = tenmat(Ax1, 1);
    Ax1 = double(Ax1);
    Ax4 = X(:, :, 2); % take the first slice of tensor X
    Ax4 = tenmat(Ax4, 1);
    Ax4 = double(Ax4);
    Ax_init = [Ax1 Ax4];
    
    % [rAx, cAx] = size(Ax_init);
    % fprintf(1, 'size of Ax slice = [%d,%d] \n', rAx, cAx);
    Ay1 = Y_org(:, :, 1); % take the first slice of tensor Y_org (the one that's not multiply by \Pi)
    Ay1 = tenmat(Ay1, 1);
    Ay1 = double(Ay1);
    Ay4 = Y_org(:, :, 2); % take the first slice of tensor Y_org (the one that's not multiply by \Pi)
    Ay4 = tenmat(Ay4, 1);
    Ay4 = double(Ay4);
    Ay_init = [Ay1 Ay4];
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