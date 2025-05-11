% CNN Model - Predicting Pollutants using Convolutional Neural Network

% Normalize inputs and outputs across samples
X = normalize(X, 2);
Y = normalize(Y, 2);

% Reshape input for CNN (Height x Width x Channels x Samples)
X = reshape(X, [8, 1, 1, 1310]);
Y = Y';

% Define CNN layers
layers = [
    imageInputLayer([8 1 1], 'Normalization', 'none')
    convolution2dLayer([3 1], 32, 'Padding', 'same')
    reluLayer()
    fullyConnectedLayer(64)
    reluLayer()
    fullyConnectedLayer(1)
    regressionLayer()
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', 3e-4, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the CNN
net = trainNetwork(X, Y, layers, options);

% Predict and reshape outputs
YPred = predict(net, X);
YPred = YPred(:); Y = Y(:);

% Evaluate performance
rmse = sqrt(mean((YPred - Y).^2));
disp(['RMSE (CNN): ', num2str(rmse)]);

% Scatter plot
figure;
scatter(Y, YPred, 'filled');
hold on;
plot([min(Y) max(Y)], [min(Y) max(Y)], 'r--', 'LineWidth', 2);
xlabel('Observed'); ylabel('Predicted');
title('CNN: Predicted vs Observed');
legend('Data', 'Perfect Fit');
grid on;