function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Input layer
a1 = X';

% Hidden layer
a1 = [ones(1, size(a1,2)) ; a1]; % Add the bias- extra row of 1's
z2 = Theta1 * a1;
a2 = sigmoid(z2);

% Output layer
a2 = [ones(1, size(a2,2)) ; a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% Cost computation

% --compute cost function J without regularization term--

for i = 1:num_labels
  yNew = (y == i);
  pNew = a3(i,:)';
  hypothesis = pNew;
  
  term1 = (-yNew' * log(hypothesis));
  term2 = (1-yNew)';
  term3 = log(1-hypothesis);

  J = J + ( (1 / m) .* ( term1 - (term2 * term3) ) );
end

  % Remove the bias (1's element)
  Theta1Orig = Theta1(:, 2:end);
  Theta2Orig = Theta2(:, 2:end);
  
  % Vectorize the matrices
  Theta1Vec = Theta1Orig(:);
  Theta2Vec = Theta2Orig(:);
  
  squared1 = Theta1Vec .^2;
  squared2 = Theta2Vec .^2;

  regularization = (lambda / (2 * m) ) * (sum(squared1) + sum(squared2));
  J = J + regularization;
  
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

D1=0;
D2=0;

for t = 1 : m

  % FORWARD PROPAGATION
  a = X(t, :); % 1 x 400
  
  % Input layer
  a1 = a'; % 400 x 1  

  % Hidden layer
  a1 = [ones(1, size(a1,2)) ; a1]; % Add the bias- extra row of 1's
  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  
  % Output layer
  a2 = [ones(1, size(a2,2)) ; a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  % BACKWARD PROPAGATION
  yElem = y(t);  % 1x1 convert to 10x1
  yMat = zeros(num_labels,1);
  yMat(yElem) = 1;
  d3 = a3 - yMat;

  % +1 Bias must be removed for Theta2
  a2Grad = sigmoidGradient(z2);
  d2 = (Theta2Orig' * d3 ) .* a2Grad;  % 25x1

  D2 = D2 + d3 * a2';  % 10x26
  D1 = D1 + d2 * a1';  % 25x401
 
end
Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) += ((lambda / m) * Theta1Orig);
Theta2_grad(:, 2:end) += ((lambda / m) * Theta2Orig);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
