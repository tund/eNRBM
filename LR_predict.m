function y = LR_predict(L, X)
%LR_PREDICT Predict the class
%

y = LR_predict_prob(L, X);	% get class probabilities
y = double(y >= 0.5);		% compare with a threshold: 0.5

end

