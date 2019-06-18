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
one1 = ones(m,1);
X = [one1,X];
z_2 = X*Theta1'; %5000 * 25
hidden = sigmoid(z_2);
one2 = ones(size(hidden,1),1);
hidden = [one2,hidden]; % hidden is 5000 * 26
z_3 = hidden * Theta2';
out = sigmoid(z_3); % out is 5000 * 10
SUM = 0;
Y = zeros(m,num_labels);
for j = 1 : m
    Y(j,y(j)) = 1;
end
% Y is 5000 * 10
for i = 1 : m
    tep = Y(i,:)*log(out(i,:)') + (1-Y(i,:))*log(1-out(i,:)');
    SUM = SUM + tep;
end

J = -SUM/m;

%=================================================

reg = lambda/2/m*(sum(Theta1(:,2:end).^2,'all') + sum(Theta2(:,2:end).^2,'all'));
J = J +reg;

%=================================================
DELTA_1 = 0;
DELTA_2 = 0;
for i = 1 : m
    delta_3 = out(i,:) - Y(i,:); % 1 * 10
    delta_2 = delta_3*Theta2(:,2:end).*sigmoidGradient(z_2(i,:)); % 1*25
    DELTA_2 = DELTA_2 + delta_3' * hidden(i,:);  % 10 * 26
    DELTA_1 = DELTA_1 + delta_2' * X(i,:); % 25 * 401
end

Theta1_grad = DELTA_1 ./ m;
Theta2_grad = DELTA_2 ./ m;

Theta1_grad = reshape(Theta1_grad,[],1);
Theta2_grad = reshape(Theta2_grad,[],1);

grad = [Theta1_grad;Theta2_grad];

%================================================================
Theta1_grad = DELTA_1 ./ m;
Theta2_grad = DELTA_2 ./ m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end).*lambda./m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end).*lambda./m;

Theta1_grad = reshape(Theta1_grad,[],1);
Theta2_grad = reshape(Theta2_grad,[],1);

grad = [Theta1_grad;Theta2_grad];


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
