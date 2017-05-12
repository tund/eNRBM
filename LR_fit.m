function [L] = LR_fit(L, X, y)
%LR_FIT Learn parameters with training data "X" and training labels "y"
%

%---- refine labels ----
[~, ~, y] = unique(y);
y(y == 1) = 0;
y(y == 2) = 1;

%---- initialize weights ----
w = zeros(size(X, 2)+1, size(y, 2));

% add 1-column to data for biases
X = [X, ones(size(X, 1), 1)];

size_w = size(w);
%---- train using LBFGS ----
[w, ~, it] = LBFGS0(@LR_grad, w(:), 10, 0.1, 1E-7, 1E-5, ...
					L.max_iter, L.verbose.iter, L, X, y);
w = reshape(w, size_w);

%---- keep learnt parameters ----
L.w = w(1:end-1, :);
L.b = w(end, :);
L.n_iter = it;
	
end

