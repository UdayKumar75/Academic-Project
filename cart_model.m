% CART Model - Regression Tree for Air Pollution Prediction

% Normalize input and output data across samples
X = normalize(X, 2);
Y = normalize(Y, 2);

% Ensure Y is a column vector
if size(Y, 2) > size(Y, 1)
    Y = Y';
end

% Train a CART regression tree
cartModel = fitrtree(X, Y, ...
    'MinLeafSize', 5, ...
    'Surrogate', 'on');

% Predict using the CART model
YPred = predict(cartModel, X);

% Evaluate model
rmse = sqrt(mean((YPred - Y).^2));
disp(['RMSE (CART): ', num2str(rmse)]);

% R-squared calculation
SSres = sum((Y - YPred).^2);
SStot = sum((Y - mean(Y)).^2);
R2 = 1 - (SSres / SStot);
disp(['R² (CART): ', num2str(R2)]);

% Scatter plot
figure;
scatter(Y, YPred, 'filled');
hold on;
plot([min(Y) max(Y)], [min(Y) max(Y)], 'r--', 'LineWidth', 2);
xlabel('Observed'); ylabel('Predicted');
title('CART: Predicted vs Observed');
legend('Data', 'Perfect Fit');
grid on;