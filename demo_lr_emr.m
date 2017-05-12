load('gen_data.mat');

N_FOLDS = numel(idx_train);
N_LABELS = numel(unique(y));

prob_train = cell(1, N_FOLDS);
prob_test = cell(1, N_FOLDS);

inst_lr_arr = cell(1, N_FOLDS);

fprintf(1, 'Running Logistic Regression...\n');

for ifold=1:N_FOLDS
	fprintf(1, '\tfold #%d...\n', ifold);

	n_feat = size(X, 2);
	n_trains = numel(idx_train{ifold});
	n_tests = numel(idx_test{ifold});
	
	X_train = X(idx_train{ifold}, :);
	y_train = y(idx_train{ifold});
	X_test = X(idx_test{ifold}, :);
	y_test = y(idx_test{ifold});
	
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

	