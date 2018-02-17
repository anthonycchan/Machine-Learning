function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis = X * theta;
squaredError = (hypothesis - y) .^2;

noBiasTheta = theta(2:end, :);
regularization = ((lambda/(2*m)) * (sum(noBiasTheta .^ 2)));

J = (1/(2*m) * sum(squaredError) ) + regularization;

% =========================================================================
% Gradient (grad contains all thetas)
regularization = (lambda / m) * theta;
grad = ( (1 / m) * ( X' * (hypothesis - y) ) ) + regularization;

% Get none regularized theta 1
term4= (hypothesis - y);
term5 = X';
term5 = term5(1,:);

theta1 = 1 / m * ( term5 * term4 );

grad(1) = theta1;

grad = grad(:);

end
