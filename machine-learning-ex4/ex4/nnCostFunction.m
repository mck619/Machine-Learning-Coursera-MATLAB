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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m,1),X];   %adds ones to first column for bias unit
X = transpose(X);    %need columns instead of rows to be each training example
z2 = Theta1*X;
a2 = sigmoid(z2);    %each column is now a2 for each training example
m1 = size(a2, 2);
a2 = [ones(1,m1);a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);
ym = zeros(num_labels,m);


for i = 1:m
    yvect = zeros(num_labels,1);
    yvect(y(i)) = 1;
    ym(:,i) = yvect;
    jt = -1*transpose(yvect)*log(a3(:,i)) - transpose((1-yvect))*log(1-a3(:,i));
    J = J + jt;
end

J = J/m;
Theta1R = Theta1(:,2:end).^2;
Theta2R = Theta2(:,2:end).^2;

reg = lambda/(2*m)*(sum(Theta1R(:)) + sum(Theta2R(:)));


J = J + reg;

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

delta_3 = a3-ym;
delta_2 = transpose(Theta2)*delta_3.*a2.*(1-a2);
DELTA_2 = transpose(a2*transpose(delta_3));
DELTA_1 = transpose(X*transpose(delta_2));
reg_t1 = [zeros(hidden_layer_size,1), lambda/m*(Theta1(:,2:end))];
reg_t2 = [zeros(num_labels,1), lambda/m*(Theta2(:,2:end))];
    
Theta1_grad = 1/m*(DELTA_1(2:end,:)) + reg_t1;
Theta2_grad = 1/m*DELTA_2 + reg_t2;


size(Theta1_grad)
size(Theta2_grad)





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
