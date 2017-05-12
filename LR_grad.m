function [f, df] = LR_grad(w, L, X, y)
%LR_GRAD Compute cost function and gradients of weights
%

% N = number of data points
% K = number of features
% C = number of labels
[N, K] = size(X);
C = size(y, 2);
w = reshape(w, [K, C]);

% compute p(y|x) basing on link function
prob = sigmoid(X*w);

% compute loss/(negative)log-likelihood
f = - y'*log(max(prob, realmin)) - (1-y)'*log(max(1-prob, realmin));

f = f ./ N;

% l2 regularization
if L.l2_penalty > 0
    w2 = w(1:end-1, :).^2;
	f = f + 0.5*L.l2_penalty*(sum(w2(:)));
end

% l1 regularization
if L.l1_penalty > 0
	w2sqrt = sqrt(w(1:end-1, :).^2 + L.l1_smooth);
	f = f + L.l1_penalty*(sum(w2sqrt(:)));    
end

if nargout > 1
    df = - X'*(y-prob);
    df = df ./ N;
    
	% l1 regularization
	if L.l2_penalty > 0
		df(1:end-1, :) = df(1:end-1, :) + L.l2_penalty*w(1:end-1, :);
	end

	% l2 regularization
	if L.l1_penalty > 0
		df(1:end-1, :) = df(1:end-1, :) + L.l1_penalty*(w(1:end-1, :)./w2sqrt);		
	end
	
	df = df(:);
end

end

