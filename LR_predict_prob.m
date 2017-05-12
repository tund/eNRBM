function y = LR_predict_prob(L, X)
%LR_PREDICT_PROB Predict the class probabilities
%

X = [X, ones(size(X, 1), 1)];
y = sigmoid(X*[L.w; L.b]);

end

