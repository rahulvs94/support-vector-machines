function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 3;
sigma = 3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

choice = [0.01 0.03 0.1 0.3 1 3 10 30];
minError = Inf;
curC = Inf;
cur_sigma = Inf;

listError = zeros(size(choice,2)*size(choice,2),1);
listC = zeros(size(choice,2)*size(choice,2),1);
listSigma = zeros(size(choice,2)*size(choice,2),1);
temp = 1;
for TempC = choice
    for TempSigma = choice
		model = svmTrain(X, y, TempC, @(x1, x2) gaussianKernel(x1, x2, TempSigma));
		prediction = svmPredict(model , Xval);
		error = mean(double(prediction ~= yval));
        listError(temp,1) = error;
        listC(temp,1) = TempC;
        listSigma(temp,1) = TempSigma;
        if (minError > error)
			minError = error;
			C = TempC;
			sigma = TempSigma;
        end
        temp = temp + 1;
    end
end

minError = min(listError);
pos = find(listError == minError);
disp(pos);
% disp(listC);
% disp(listSigma);
disp(listError);

% if (numel(pos)>1)
%     fprintf('Number of elements for which error was similar(minimum): %d \n\n', numel(pos));
%     TempGreaterC = zeros(numel(pos),1);
%     TempGreSigma = zeros(numel(pos),1);
%     for i = pos
%         TempGreaterC(i) = listC(i,1);
%         TempGreSigma(i) = listSigma(i,1);
%     end
%     C = max(TempGreaterC);
%     f = TempGreaterC == C;
%     sigma = TempGreSigma(f,1);
%     %%%%% need to implement for sigma (if sigma values are different for same C)
% else
%     fprintf('Only once the error was minimum out of %d models\n',(numel(choice)*numel(choice)));    
%     fprintf('Minimum error: %.3f \n', minError);
%     C = listSigma(pos,1);
%     sigma = listC(pos,1);
%     fprintf('Sigma: %.3f \nC: %.3f \n', listSigma(pos,1),listC(pos,1));
% end

% =========================================================================

end
