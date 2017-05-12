% Evaluate the performance by computing "Recall", "Precision", "F1" for each risk class
fprintf('Evaluating...\n');

isCalibration 	= 1;

lowerPercentile	= 78;
upperPercentile	= 93;

L = 3;
C1 = 1; C2 = 2; C3 = 3;

clear predictStateFolds;
clear trueC2Ids;
clear trueC3Ids;

clear noRiskRecalls;
clear noRiskPrecs;
clear noRiskF1s;

clear lowRiskRecalls;
clear lowRiskPrecs;
clear lowRiskF1s;

clear highRiskRecalls;
clear highRiskPrecs;
clear highRiskF1s;

testStatesAll 		= [];
predictStatesAll 	= [];
predictProbsAll 	= [];

trainLL = 0;
testLL	= 0;
currTrainSize	= 0;
currTestSize	= 0;

testExpectedRiskAll = [];
testIdsAll			= [];

noRiskRecalls = zeros(1, N_FOLDS);
noRiskPrecs	= zeros(1, N_FOLDS);
noRiskF1s = zeros(1, N_FOLDS);

lowRiskRecalls = zeros(1, N_FOLDS);
lowRiskPrecs = zeros(1, N_FOLDS);
lowRiskF1s = zeros(1, N_FOLDS);

highRiskRecalls = zeros(1, N_FOLDS);
highRiskPrecs = zeros(1, N_FOLDS);
highRiskF1s = zeros(1, N_FOLDS);

for k=1:N_FOLDS

	testIdsAll = [testIdsAll; idx_test{k}];

	trainStates	= y(idx_train{k});
	testStates 	= y(idx_test{k});

	trainSize 	= length(trainStates);
	testSize	= length(testStates);

	currTrainSize	= currTrainSize + trainSize;
	currTestSize	= currTestSize + testSize;

	% -- estimate the risk states in the test data --
	epsilon = 1e-4;

	for i=1:trainSize
		prob = (1-epsilon)*prob_train{k}(i, trainStates(i)) + epsilon/L;
		trainLL = trainLL + log(prob);
	end
	for i=1:testSize
		prob = (1-epsilon)*prob_test{k}(i, testStates(i)) + epsilon/L;
		testLL = testLL + log(prob);
	end

	[maxVals, maxStates] = max(prob_test{k}');
	predictStates = maxStates';

	trainExpectedRisks	= prob_train{k} * [0:2]';
	testExpectedRisks 	= prob_test{k} * [0:2]';

	testExpectedRiskAll = [testExpectedRiskAll; testExpectedRisks];

	predictProbsAll = [predictProbsAll; prob_test{k}];

	% ------- CALIBRATION ------------------

	if isCalibration

		noRiskMean		= mean(trainExpectedRisks);
		noRiskStd 		= std(trainExpectedRisks);
		lowRiskMean		= mean(trainExpectedRisks(trainStates >= 2));
		lowRiskStd		= std(trainExpectedRisks(trainStates >= 2));

		highRiskMean	= mean(trainExpectedRisks(trainStates == 3));
		highRiskStd 	= std(trainExpectedRisks(trainStates == 3));

		threshold1 = prctile(trainExpectedRisks, lowerPercentile);
		threshold2 = prctile(trainExpectedRisks, upperPercentile);

		predictStates(testExpectedRisks <= threshold1) 	= C1;
		predictStates(testExpectedRisks > threshold1)	= C2;
		predictStates(testExpectedRisks > threshold2)	= C3; %overriding the low-risk states
	end
	
	predictStateFolds{k} = predictStates;

	testStatesAll		= [testStatesAll; testStates];
	predictStatesAll	= [predictStatesAll; predictStates];


	confuseMatrix = zeros(L, L);
	for l=1:L
		for l2=1:L
			confuseMatrix(l, l2) = sum(testStates==l & predictStates==l2);
		end
	end

	noRiskRecall	= 100*confuseMatrix(1,1)/(1e-10 + sum(confuseMatrix(1,:)));
	noRiskPrec		= 100*confuseMatrix(1,1)/(1e-10 + sum(confuseMatrix(:,1)));
	noRiskF1		= 2*noRiskRecall*noRiskPrec/(1e-10 + noRiskRecall + noRiskPrec);

	lowRiskRecall	= 100*confuseMatrix(L-1,L-1)/(1e-10 + sum(confuseMatrix(L-1,:)));
	lowRiskPrec		= 100*confuseMatrix(L-1,L-1)/(1e-10 + sum(confuseMatrix(:,L-1)));
	lowRiskF1		= 2*lowRiskRecall*lowRiskPrec/(1e-10 + lowRiskRecall + lowRiskPrec);

	highRiskRecall	= 100*confuseMatrix(L,L)/(1e-10 + sum(confuseMatrix(L,:)));
	highRiskPrec	= 100*confuseMatrix(L,L)/(1e-10 + sum(confuseMatrix(:,L)));
	highRiskF1		= 2*highRiskRecall*highRiskPrec/(1e-10 + highRiskRecall + highRiskPrec);

	noRiskRecalls(k)	= noRiskRecall;
	noRiskPrecs(k)		= noRiskPrec;
	noRiskF1s(k)		= noRiskF1;

	lowRiskRecalls(k)	= lowRiskRecall;
	lowRiskPrecs(k)		= lowRiskPrec;
	lowRiskF1s(k)		= lowRiskF1;

	highRiskRecalls(k)	= highRiskRecall;
	highRiskPrecs(k)	= highRiskPrec;
	highRiskF1s(k)		= highRiskF1;
end


trainLL	= trainLL / currTrainSize;
testLL	= testLL / currTestSize;

%print out the overall confusion matrix
confuseMatrixAll = zeros(L,L);
suicideVectorAll = zeros(1,L);
for l=1:L
	for l2=1:L
		confuseMatrixAll(l,l2) = sum(testStatesAll==l & predictStatesAll==l2);
	end
end

trueC2Ids = testIdsAll(testStatesAll==2 & predictStatesAll==2);
trueC3Ids = testIdsAll(testStatesAll==3 & predictStatesAll==3);

noRiskRecall	= 100*confuseMatrixAll(1,1)/(1e-10 + sum(confuseMatrixAll(1,:)));
noRiskPrec		= 100*confuseMatrixAll(1,1)/(1e-10 + sum(confuseMatrixAll(:,1)));
noRiskF1		= 2*noRiskRecall*noRiskPrec/(1e-10 + noRiskRecall + noRiskPrec);

lowRiskRecall	= 100*confuseMatrixAll(L-1,L-1)/(1e-10 + sum(confuseMatrixAll(L-1,:)));
lowRiskPrec		= 100*confuseMatrixAll(L-1,L-1)/(1e-10 + sum(confuseMatrixAll(:,L-1)));
lowRiskF1		= 2*lowRiskRecall*lowRiskPrec/(1e-10 + lowRiskRecall + lowRiskPrec);

highRiskRecall	= 100*confuseMatrixAll(L,L)/(1e-10 + sum(confuseMatrixAll(L,:)));
highRiskPrec	= 100*confuseMatrixAll(L,L)/(1e-10 + sum(confuseMatrixAll(:,L)));
highRiskF1		= 2*highRiskRecall*highRiskPrec/(1e-10 + highRiskRecall + highRiskPrec);

% save to variable
clear result;
result.m = zeros(3,3);
result.m(1,1) = mean(noRiskRecalls);
result.m(1,2) = mean(noRiskPrecs);
result.m(1,3) = mean(noRiskF1s);
result.m(2,1) = mean(lowRiskRecalls);
result.m(2,2) = mean(lowRiskPrecs);
result.m(2,3) = mean(lowRiskF1s);
result.m(3,1) = mean(highRiskRecalls);
result.m(3,2) = mean(highRiskPrecs);
result.m(3,3) = mean(highRiskF1s);

result.s = zeros(3,3);
result.s(1,1) = std(noRiskRecalls);
result.s(1,2) = std(noRiskPrecs);
result.s(1,3) = std(noRiskF1s);
result.s(2,1) = std(lowRiskRecalls);
result.s(2,2) = std(lowRiskPrecs);
result.s(2,3) = std(lowRiskF1s);
result.s(3,1) = std(highRiskRecalls);
result.s(3,2) = std(highRiskPrecs);
result.s(3,3) = std(highRiskF1s);

result.m = result.m / 100;
result.s = result.s / 100;

fprintf(1, '%-16s %-16s %-16s %-16s\n', 'Risk Level', 'Recall', 'Precision', 'F1');
risk_name = {'No risk', 'Medium Risk', 'High Risk'};
for i=1:3
	fprintf(1, '%-16s', risk_name{i});
	for j=1:2
		fprintf(1,'%.3f+-%.3f\t', result.m(i,j), result.s(i,j));
	end
	j=3;
	fprintf(1,'%.3f+-%.3f\n', result.m(i,j), result.s(i,j));
end
