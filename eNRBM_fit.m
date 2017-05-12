function [R] = eNRBM_fit(R, X, varargin)
%eNRBM_FIT Train eNRBM
%

%% parse input arguments
parser = inputParser;
addRequired(parser, 'X', @isnumeric);
addOptional(parser, 'to_restart', true, @islogical);	% restart or continue training

parse(parser, X, varargin{:});

%% assign
to_restart = parser.Results.to_restart;

K = R.n_hid;
[n_data, N] = size(X);
bsize = R.batch_size;

%% initialize parameters
if to_restart			% restart training -> re-initialize parameters
	h = zeros(1, K);
	v = zeros(1, N);
	w = zeros(N, K);
	hgrad_inc = zeros(1, K);
	vgrad_inc = zeros(1, N);
	wgrad_inc = zeros(N, K);
	R.n_iter = 0;
	R.recon_err = zeros(R.max_iter, 1);
    R.train_time = 0;
	
	R.momentum.b = R.momentum.b_init;
	R.momentum.w = R.momentum.w_init;
	R.lrate.h0 = R.lrate.h;
	R.lrate.v0 = R.lrate.v;
	R.lrate.w0 = R.lrate.w;
else	% continue training -> take current parameters
	h = R.h;
    v = R.v;
    w = R.w;
    hgrad_inc = R.hgrad_inc;
    vgrad_inc = R.vgrad_inc;
    wgrad_inc = R.wgrad_inc;
	if R.n_iter < R.max_iter
		tmp = R.recon_err;
		R.recon_err = zeros(R.max_iter, 1);
		R.recon_err(1:length(tmp)) = tmp;
	end
end

%% ---- train ----
fprintf(1, '%10s %15s %15s\n', 'Iteration', 'Error', 'Time');

start = tic();

for iter=R.n_iter+1:R.max_iter
	
	R.recon_err(iter) = 0;
	
	for ibatch=1:bsize:n_data
        ibatch_max = min(n_data, ibatch+bsize-1);
        ibsize = ibatch_max-ibatch+1;                % number of samples in R mini-batch
        batch = X(ibatch:ibatch_max,:);              % data mini-batch
		
		% initialize gradients
        pos_hgrad = zeros(1, K);
        pos_vgrad = zeros(1, N);
        pos_wgrad = zeros(N, K);        
		
		%---- clamp phase ----
		hprob = sigmoid(bsxfun(@plus, batch*w, h));		% compute hidden probabilities
		
		% sparsity of the hidden
		if R.sparse_weight > 0
			if ibatch > 1 && ibsize == bsize
				q = R.sparse_decay*prev_hprob + (1-R.sparse_decay)*hprob;
			else
				q = (1-R.sparse_decay)*hprob;
			end
			prev_hprob = hprob;
			
			sparse_grad = R.sparse_level - q;
			pos_hgrad = pos_hgrad + R.sparse_weight*mean(sparse_grad);
			pos_wgrad = pos_wgrad + R.sparse_weight*((batch'*sparse_grad)/ibsize);
		end
				
		pos_hgrad = pos_hgrad + mean(hprob);
		pos_vgrad = pos_vgrad + mean(batch);
		pos_wgrad = pos_wgrad + (batch'*hprob)/ibsize;
		
		%---- free phase ----
		% perform (n_cd) contrastive divergence steps
		for icd=1:R.learning.n_cd
			hsample = double(rand(ibsize, K) < hprob);		% sample hidden units
			vprob = sigmoid(bsxfun(@plus, hsample*w', v));	% compute visible probabilities
			vsample = double(rand(ibsize, N) < vprob);		% sample visible units
			hprob = sigmoid(bsxfun(@plus, vsample*w, h));	% compute hidden probabilities
		end
		
		%---- negative phase ----
		neg_hgrad = mean(hprob);
		neg_vgrad = mean(vsample);
		neg_wgrad = (vsample'*hprob) / ibsize;
		
        %---- update gradients ----
        hgrad_inc = R.momentum.b*hgrad_inc + R.lrate.h*(pos_hgrad-neg_hgrad);
        vgrad_inc = R.momentum.b*vgrad_inc + R.lrate.v*(pos_vgrad-neg_vgrad);
        wgrad_inc = R.momentum.w*wgrad_inc + R.lrate.w*(pos_wgrad-neg_wgrad - R.wc*w);
		
		%---- nonnegative restriction ----
		idx = (w<0);
		wgrad_inc(idx) = wgrad_inc(idx) - R.lrate.w*R.nonneg_cost*w(idx);
		
		%---- smoothness ----
		if R.smooth_cost > 0
			wgrad_inc = wgrad_inc - R.smooth_cost * (R.correl*w);
		end
		
		%---- update parameters ----
        h = h + hgrad_inc;
        v = v + vgrad_inc;
        w = w + wgrad_inc;
		
		%---- compute reconstruction error ----
		R.recon_err(iter) = R.recon_err(iter) + sum(sum(abs(batch-vsample)));		
	end
	
	R.recon_err(iter) = R.recon_err(iter) / (n_data*N);
	
	fprintf(1, '%10d %15.4f %15.4f\n', iter, R.recon_err(iter), toc(start));
	
	%---- adjust momentums ----
	if iter >= R.momentum.iter
		R.momentum.b = R.momentum.b_final;
		R.momentum.w = R.momentum.w_final;
	end
	
	%---- adjust learning rates ----
	R.lrate.h = R.lrate.h0 / sqrt(iter);
	R.lrate.v = R.lrate.v0 / sqrt(iter);
	R.lrate.w = R.lrate.w0 / sqrt(iter);
end

R.train_time = toc(start);

% store parameters
R.n_iter = iter;
R.h = h;
R.v = v;
R.w = w;
R.hgrad_inc = hgrad_inc;
R.vgrad_inc = vgrad_inc;
R.wgrad_inc = wgrad_inc;

end

