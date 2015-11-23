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

% Convert y to a matrix using logical arrays
temp_y = zeros(m,num_labels);
for i = 1:m;
temp_y(i,:) = 1:num_labels;
endfor

temp_y = (temp_y == y);
y = temp_y ;

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% computing forward propagation
a1 = [ones(m, 1) X] ;
z2 = a1*Theta1' ;
a2 = sigmoid(z2) ;
a2 = [ones(m, 1) a2];
z3 = a2*Theta2' ;
a3 = sigmoid(z3) ; 
hx = a3 ;
 

% computing the cost function
A = 0;
for i = 1:m;
    for k = 1:num_labels;
        A = A + (-y(i,k) * log(hx(i,k)) - (1 - y(i,k)) * log(1 - hx(i,k))) ;
    endfor
endfor
J = (1/m) * A;

% computing regularized cost function
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

%initialize delta_2 and delta_3
delta_2 = zeros(m,num_labels);
delta_3 = zeros(m,num_labels);


% computing back propagation
for t = 1:m;
	a1 = [ 1 ; X(t,:)'] ;
	% forward propagation
	z2 = Theta1 * a1 ; 
	a2 = [1 ; sigmoid(z2)] ; 
	z3 = Theta2 * a2 ;
	a3 = sigmoid(z3) ; 

	% the difference between the expected output and the current output
	d3 = a3 - y(t,:)' ; 

	z2 = [1 ; z2] ; 
	% for hidden layer
	d2 = Theta2' * d3 .* sigmoidGradient(z2) ;
	d2 = d2(2:end) ;

	Theta1_grad = (Theta1_grad + d2 * a1') ;
	Theta2_grad = (Theta2_grad + d3 * a2') ;

endfor

% unregularized gradient fot nn cost function
Theta1_grad = Theta1_grad / m ;
Theta2_grad = Theta2_grad / m ;


% regularization added to the gradient computed above
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:,2:end) ;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:,2:end) ;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
