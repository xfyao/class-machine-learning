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

H = sigmoid(X*theta);

J = sum((-1)*y.*log(H) + (-1)*(1-y).*log(1-H))/m + (sum(theta.^2)-theta(1)^2)*lambda/(2*m);

n = size(theta);


grad = (X'*(H-y))/m + theta*(lambda/m);
grad(1) = (X'(1,:)*(H-y))/m;
%grad([2,n]) = (X'([2,n],:)*(H-y))/m + theta([2,n])*(lambda/m);





% =============================================================

end
