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

result = sigmoid(X * theta);
reg = theta' * theta - theta(1)^2;
J = (-1 * y' * log(result) - (1 - y)' * log(1 - result)) / m;
J = J + lambda * reg / (2 * m);

for i = 1 : size(theta)
    if i == 1
        grad(i) = (result - y)' * X(:, i) / m;
    end
    if i != 1
        grad(i) = (result - y)' * X(:,i) / m + theta(i) * lambda / m;
end


% =============================================================

end
