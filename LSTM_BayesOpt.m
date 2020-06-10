function [RMSE_ave] = LSTM_BayesOpt(XTrain,YTrain,XVal,YVal,bayes_learnrate,bayes_hiddenunit,bayes_fullyconnected,bayes_dropout)
% Function to calculate the average RMSE for bayesian optimisation of LSTM
% model

% state the inputs
numFeatures = 6;
numResponses = 1;
optimal_ep = 100;
miniBatchSize = 10;

% for reproducability 
rng('default')

% options for LSTM training
options = trainingOptions('adam', ...
                    'MaxEpochs',optimal_ep, ...
                    'MiniBatchSize',miniBatchSize, ...
                    'ValidationData',{XVal,YVal},...
                    'ValidationFrequency',25,...
                    'ValidationPatience',20,... %for early stopping
                    'GradientThreshold',1, ...
                    'LearnRateSchedule','piecewise',...
                    'InitialLearnRate',bayes_learnrate, ...
                    'Verbose',1, ...
                    'Shuffle','never');

%LSTM layer 
layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(bayes_hiddenunit,'OutputMode','sequence')
            fullyConnectedLayer(bayes_fullyconnected)
            dropoutLayer(bayes_dropout)
            fullyConnectedLayer(numResponses)
            regressionLayer];

   
% Train LSTM network        
net = trainNetwork(XTrain,YTrain,layers,options);

%prediction on the validation set 
pred_test = predict(net,XVal);
% calculate RMSE 
RMSE_val = sqrt(mean((YVal-pred_test).^2));
% calculate average RMSE
RMSE_ave = mean(RMSE_val);


end

