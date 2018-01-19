function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Cost
hypothesis = sigmoid(X * theta);

term1 = (-y' * log(hypothesis));
term2 = (1-y)';
term3 = log(1-hypothesis);


squared= theta(2:length(theta)) .^ 2;
regularization = (lambda / (2 * m) ) * sum(squared);

J = ( (1 / m) .* ( term1 - (term2 * term3) ) ) + regularization;

% Gradient (grad contains all thetas)
regularization = (lambda / m) * theta;
grad = ( (1 / m) * ( X' * (hypothesis - y) ) ) + regularization;

% Get none regularized theta 1
term4= (hypothesis - y);
term5 = X';
term5 = term5(1,:);

theta1 = 1 / m * ( term5 * term4 );

grad(1) = theta1;

% =============================================================

end
