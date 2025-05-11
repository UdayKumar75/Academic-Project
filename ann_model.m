% ANN Model - Predicting Air Pollutants using Feedforward Neural Network

% Normalize input data across features
X = normalize(X, 1);

% Ensure Y is a column vector
if size(Y, 2) > size(Y, 1)
    Y = Y';
end

% Define a feedforward neural network with two hidden layers
hiddenLayerSizes = [20 10];
net = fitnet(hiddenLayerSizes, 'trainbr'); % 'trainbr' uses Bayesian Regularization

% Set training parameters
net.trainParam.showWindow = false;
net.trainParam.epochs = 500;
net.performFcn = 'mse';

% Data division ratios
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train the network
net = train(net, X', Y');

% Make predictions
YPred = net(X')';

% Evaluate model
rmse = sqrt(mean((YPred - Y).^2));
disp(['RMSE (ANN): ', num2str(rmse, '%.4f')]);

% R-squared calculation
SSres = sum((Y - YPred).^2);
SStot = sum((Y - mean(Y)).^2);
R2 = 1 - (SSres / SStot);
disp(['R² (ANN): ', num2str(R2, '%.4f')]);

% Scatter plot
figure;
scatter(Y, YPred, 'filled');
hold on;
plot([min(Y) max(Y)], [min(Y) max(Y)], 'r--', 'LineWidth', 2);
xlabel('Observed'); ylabel('Predicted');
title('ANN: Predicted vs Observed');
legend('Data', 'Perfect Fit');
grid on;