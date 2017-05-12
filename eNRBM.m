function [R] = eNRBM()
%eNRBM Create a default "EMR-driven Nonnegative Restricted Boltzmann Machine (eNRBM)"
% All parameters are set by default

R = [];
R.n_hid = 100;				% number of hidden units
R.batch_size = 100;			% number of data points in a minibatch
R.max_iter = 100;			% maximum number of iterations

R.wc = 0.0001;				% weight decay (l2 regularization)
R.sparse_weight = 0.0;		% sparsity penalty of hidden activations
R.sparse_level = 0.1;		% sparsity level of hidden activations
R.sparse_decay = 0.9;		% sparsity decay

R.learning.n_cd = 1;		% number of Contrastive Divergence (CD)

R.momentum.b = 0.0;			% momentum of biases
R.momentum.w = 0.0;			% momentum of weights
R.momentum.iter = 5;		% momentum changes at #iteration
R.momentum.b_init = 0.5;	% value when learning starts
R.momentum.w_init = 0.5;	% value when learning starts
R.momentum.b_final = 0.9;	% value after the [R.momentum.iter] iterations
R.momentum.w_final = 0.9;	% value after the [R.momentum.iter] iterations

R.lrate.h = 0.1;			% learning rate of hidden biases
R.lrate.h0 = 0.1;			% learning rate of hidden biases at the beginning
R.lrate.v = 0.1;			% learning rate of visible biases
R.lrate.v0 = 0.1;			% learning rate of visible biases at the beginning
R.lrate.w = 0.1;			% learning rate of connection weights
R.lrate.w0 = 0.1;			% learning rate of connection weights at the beginning

R.nonneg_cost = 0.1;		% nonnegative penalty (alpha)
R.smooth_cost = 0.01;		% smooth penalty (lambda)
R.correl = [];				% correlation matrix

end

