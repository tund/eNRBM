load('gen_data.mat');

N_FOLDS = numel(idx_train);
N_LABELS = numel(unique(y));

inst_enrbm0 = eNRBM();
inst_enrbm0.n_hid = 200;
inst_enrbm0.max_iter = 10;
inst_enrbm0.learning.n_cd = 1;
inst_enrbm0.batch_size = 100;
inst_enrbm0.init_w = 0.1;

% learning rate
inst_enrbm0.lrate.h = 0.1;
inst_enrbm0.lrate.v = 0.1;
inst_enrbm0.lrate.w = 0.1;

inst_enrbm0.momentum.iter = 5;

inst_enrbm0.wc = 2E-4;
inst_enrbm0.sparse_weight = 0.0;

inst_enrbm0.nonneg_cost = 0.01;

%---- smoothness ----
inst_enrbm0.smooth_cost = 0.001;
correl = spconvert(load('feat_correl.txt'));
n_feat = size(correl, 1);
correl(n_feat, n_feat) = 1E-10;    % to make sure that the matrix is full
inst_enrbm0.correl = full(correl(1:5321, 1:5321));
		
prob_train = cell(1, N_FOLDS);
prob_test = cell(1, N_FOLDS);

inst_enrbm_arr = cell(1, N_FOLDS);
inst_lr_arr = cell(1, N_FOLDS);

fprintf(1, 'Running Logistic Regression...\n');

for ifold=1:N_FOLDS
	fprintf(1, '\tfold #%d...\n', ifold);
	
	X_train = X(idx_train{ifold}, :);
	y_train = y(idx_train{ifold});
	X_test = X(idx_test{ifold}, :);
	y_test = y(idx_test{ifold});
	
	inst_enrbm = inst_enrbm0;
	inst_enrbm = eNRBM_fit(inst_enrbm, X_train);
	inst_enrbm_arr{ifold} = inst_enrbm;
	
	X_train = eNRBM_transform(inst_enrbm, X_train);
	X_test = eNRBM_transform(inst_enrbm, X_test);

	n_feat = size(X, 2);
	n_trains = numel(idx_train{ifold});
	n_tests = numel(idx_test{ifold});
	
	prob_train{ifold} = zeros(n_trains, 3);
	prob_test{ifold} = zeros(n_tests, 3);
	
	inst_lr_arr{ifold} = [];

	for ilabel=1:N_LABELS
		fprintf(1, '\t\tlabel #%d...\n', ilabel);

		clear inst_lr;
		inst_lr = LR();
		inst_lr.l1_penalty = 0.001;
		inst_lr.l2_penalty = 0.0;
		inst_lr.verbose.iter = 1;
		
		inst_lr = LR_fit(inst_lr, X_train, y_train==ilabel);
		
		prob_train{ifold}(:, ilabel) = LR_predict_prob(inst_lr, X_train);
		prob_test{ifold}(:, ilabel) = LR_predict_prob(inst_lr, X_test);
		inst_lr_arr{ifold}{ilabel} = inst_lr;
	end

	% sum to 1
	prob_train{ifold} = bsxfun(@rdivide, prob_train{ifold}, sum(prob_train{ifold}, 2));
	prob_test{ifold} = bsxfun(@rdivide, prob_test{ifold}, sum(prob_test{ifold}, 2));
end

evaluate;

	