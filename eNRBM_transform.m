function hpost = eNRBM_transform(R, X)
%eNRBM_TRANSFORM Transform the data into hidden posteriors
%

hpost = sigmoid(bsxfun(@plus, X*R.w, R.h));

end

