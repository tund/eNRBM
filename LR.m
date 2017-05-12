function [L] = LR()
%LR Create a default "Logistic Regression (LR)"
% All parameters are set by default

L = [];
L.max_iter = 1000;		% maximum number of iterations
L.n_iter = 0;			% number of iterations which actually runs
L.l1_penalty = 0.0;		% Lasso regularization
L.l1_smooth = 1E-10;
L.l2_penalty = 0.0;     % Ridge regularization
L.w = [];				% weights
L.b = [];				% bias
L.verbose.iter = 1;		% print information during training every [L.verbose.iter] iteration(s)

end

