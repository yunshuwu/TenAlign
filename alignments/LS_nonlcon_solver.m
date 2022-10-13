function [x] = LS_nonlcon_solver(Ax, Ay, lambda1, lambda2, lambda3, x0_init)
    % return the permutation matrix/probability distribution matrix x 
    %% generate an almost identity matrix
    [row, col] = size(Ay); % row == sz(1), need to clear code here
    I_appr = eye(row, row);
    % offsets = 0.005 * rand(row, row);
    % I_appr = I_appr + offsets;
    % % normalize matrix to sumRow(M)=1
    % sumRow = sum(I_appr, 2); % sum of each row
    % sumRow = sumRow.';
    % I_appr = bsxfun(@rdivide, I_appr, sumRow(:)); % divide each row of the matrix with corresponding element in row vector sumRow
    % % show the approximate identity matrix
    % I_appr

    %% try fmincon(), deal with nonlinear constraints
    fun = @(x)norm(Ax - x*Ay, "fro")^2 + lambda1 * norm(I_appr - x*(x.'), "fro")^2 + lambda2 * norm(I_appr - (x.')*x, "fro")^2 %+ lambda3 * norm(x, 1);
    % fun = @(x) (-trace(x * Ay * (Ax.'))) + lambda1 * norm(I_appr - x*(x.'), "fro")^2 + lambda2 * norm(I_appr - (x.')*x, "fro")^2;
    % fun = @(x) (-trace(x * Ay * (Ax.')));

    %% Constraints
    % Pi(i, j) \in [0,1]
    lb = zeros(row, row);
    ub = ones(row, row);
    % ColSum = 1
    A = zeros(row, row*row);
    b = ones(row, 1);
    for i = 1:row
        for j = (i-1)*row+1 : i*row
            A(i, j) = 1;
        end
    end
    % A = [];
    % b = [];
    % RowSum=1, vecX = x(:), then apply Aeq*x=beq
    Aeq = zeros(row, row*row);
    beq = ones(row, 1);
    for i = 1:row
        for j = 0:(row-1)
            Aeq(i, j*row+i) = 1;
        end
    end
    x0 = ones(row, row) * x0_init;
    nonlcon = @nonlin_con;
    stopTol = 1e-10;

    % %% setting parallel pool!
    % if max(size(gcp)) == 0 % parallel pool needed
    %     parpool(maxNumCompThreads) % create the parallel pool
    % end

    options = optimoptions('fmincon','Display','final','Algorithm','sqp','TolX',stopTol);%, 'UseParallel',true);
    % options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');

    startTime = tic;
    x = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    time_fmincon_parallel = toc(startTime);
    fprintf('Parallel FMINCON optimization takes %g seconds.\n',time_fmincon_parallel);

    function [c, ceq] = nonlin_con(x)
        % Nested function, non-linear constraints
        % Input: variable to be solved, x
        % Output: two functions c and ceq
            [sz,~] = size(x);
            %% calculate Aeq and beq, rowSum=1
            % beq_ = ones(sz, 1);
            % ceq = sum(x, 2) - beq_;

            %% Apply entropy constraint, then what is the entropy function
            b_decimal = 0.5; % should be a small decimal near zero
            b_ = ones(sz, 1); % store the max value of the entropy of each row
            b_ = b_ * b_decimal; 
            % log2(x)
            c = sum(x .* log2(x), 2) - b_; % don't use log2() because it will give NaN value
            % c = [];

            ceq = [];

            %% calculate norm-1 constraint for elements of x, see if can converge to some feasible point
            % vecX = x(:);
            % b_hyperpara = 10; % play with this hyper-parameter
            % ceq = norm(vecX, 0.1) - b_hyperpara;
            % ceq = [];

            % lg2x = log2(x) % element-wise log2 of the matrix x
            % XlogX = x .* log2(x);
            % vecXlogX = XlogX(:);
            % % calculate the row sum of the matrix x*log2x
            % A_ = zeros(sz, sz*sz);
            % for i = 1:sz
            %     for j = 0:(sz-1)
            %         A_(i, j*sz+i) = 1;
            %     end
            % end
            % c = A_ * vecXlogX - b_;

            %% Leave the col-sum constraint along, too tight for relaxed permutation calculation
            % %% Apply col-sum=1 constraint
            % vecX = x(:);
            % b_col = 1;
            % A_ = zeros(sz, sz*sz);
            % b_ = ones(sz, 1) * b_col;
            % for i = 1:sz
            %     for j = (i-1)*sz+1 : i*sz
            %         A_(i, j) = 1;
            %     end
            % end
            % ceq = A_*vecX - b_;
    end

end