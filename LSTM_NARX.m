%% Read in data
m = readtable('SP500.csv');
% data from 1/1/2000 to 1/1/2020

%% Remove first column

m(:,1) = [];

%% Sort table by date

mn = sortrows(m,'Date','ascend');

%% plot data 

figure
plot(m.Date, m.Open)
title('Opening price of S&P 500 from 1st Jan 2000 to 1st Jan 2020')
xlabel('Date')
ylabel('Opening Price')

%% Descriptive statistics of Numerical Variables

HighMean = mean(m.High);
HighStd = std(m.High);
HighMin = min(m.High);
HighMax = max(m.High);
HighSkew = skewness(m.High);

LowMean = mean(m.Low);
LowStd = std(m.Low);
LowMin = min(m.Low);
LowMax = max(m.Low);
LowSkew = skewness(m.Low);

OpenMean = mean(m.Open);
OpenStd = std(m.Open);
OpenMin = min(m.Open);
OpenMax = max(m.Open);
OpenSkew = skewness(m.Open);

CloseMean = mean(m.Close);
CloseStd = std(m.Close);
CloseMin = min(m.Close);
CloseMax = max(m.Close);
CloseSkew = skewness(m.Close);

VolumeMean = mean(m.Volume);
VolumeStd = std(m.Volume);
VolumeMin = min(m.Volume);
VolumeMax = max(m.Volume);
VolumeSkew = skewness(m.Volume);

AdjCloseMean = mean(m.AdjClose);
AdjCloseStd = std(m.AdjClose);
AdjCloseMin = min(m.AdjClose);
AdjCloseMax = max(m.AdjClose);
AdjCloseSkew = skewness(m.AdjClose);


%% call function inputData

% The function returns the train, validation and test data

[raw_data, train_T, train_I, test_T, test_I, validation_T, validation_I] = inputData()

%%  Reformat data 
XTrain = train_I;
XTest = test_I;
XVal = validation_I;

YTrain = train_T;
YTest = test_T;
YVal = validation_T;

%% Baseline LSTM network 

tic 
numFeatures = 6; %num inputs
numResponses = 1; % num outputs 
maxEpochs = 100;
miniBatchSize = 10;
initialHiddenUnits = 50;


    layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(initialHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
            'MaxEpochs',maxEpochs, ...
            'MiniBatchSize',miniBatchSize, ...
            'ValidationData',{XVal,YVal},...
            'ValidationFrequency',25,...
            'ValidationPatience',20,...
            'GradientThreshold',1, ...
            'LearnRateSchedule','piecewise',...
            'InitialLearnRate',0.001, ...
            'Verbose',1, ...
            'Shuffle','never');

[net,info] = trainNetwork(XTrain,YTrain,layers,options);

pred_test_base = predict(net,XVal);

rmse_base = sqrt(mean((pred_test_base-YVal).^2))

time_base = toc;

disp("Base Line LSTM RMSE, the result is:")
disp(rmse_base)

disp("Base Line LSTM time taken, the result is:")
disp(time_base)


%% Hyper parameter optimisation


%% Epochs optimisation

miniBatchSize = 10;

optimalEpochs = [10 25 50 75 100];

rng('default')
N = 2;
err = zeros(N,1);
time = zeros(N,1);
Table_ep = [];
count = 1;

for e =1 :length(optimalEpochs)
    RMSE_Val_collect_ep = [];
    time_taken_collect_ep = [];
    for n = 1:N
        
        
        options = trainingOptions('adam', ...
                'MaxEpochs',optimalEpochs(e), ...
                'MiniBatchSize',miniBatchSize, ...
                'ValidationData',{XVal,YVal},...
                'ValidationFrequency',25,...
                'ValidationPatience',20,...
                'GradientThreshold',1, ...
                'LearnRateSchedule','piecewise',...
                'InitialLearnRate',0.001, ...
                'Verbose',1, ...
                'Shuffle','never');
        
        
        tic
        layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(initialHiddenUnits,'OutputMode','sequence')
        fullyConnectedLayer(numResponses)
        regressionLayer];
 
        net = trainNetwork(XTrain,YTrain,layers,options);

        pred_test = predict(net,XVal);
        RMSE_val = sqrt(mean((YVal-pred_test).^2));
        

        time_taken = toc;
        
        time_taken_collect_ep = [time_taken_collect_ep time_taken];

        RMSE_Val_collect_ep = [RMSE_Val_collect_ep RMSE_val];
    end
    meanRMSEVal = mean(RMSE_Val_collect_ep);

    meanTime = mean(time_taken_collect_ep);

    Table_ep(count,1) = optimalEpochs(e);
    Table_ep(count,2) = meanRMSEVal;
    Table_ep(count,3) = meanTime;
    count = count + 1;
    
end

%% Find Optimal Epochs

epochs = Table_ep(:,1);

Rmse_ep = Table_ep(:,2);

Time_graph_ep = Table_ep(:,3);

%Plot graph

plot3(epochs, Rmse_ep, Time_graph_ep, 'LineWidth', 2);
xlabel('Epochs');
ylabel('RMSE');
zlabel('Elapsed Time (seconds)');
title('Epochs Optimisation');
grid on

% Choose the optimal value 
sortTable_ep = table(epochs,Rmse_ep,Time_graph_ep);
sortTable_ep = sortrows(sortTable_ep,'Rmse_ep','ascend');

optimal_ep= sortTable_ep(1, 1);
optimal_ep = table2array(optimal_ep);
 
disp("Optimal Epochs is:")
disp(optimal_ep)

%% Hidden units Optimisation

optimal_ep = 100;

miniBatchSize = 10;

numHiddenUnits = [5 50 100 200 300 400];


options = trainingOptions('adam', ...
            'MaxEpochs',optimal_ep, ...
            'MiniBatchSize',miniBatchSize, ...
            'ValidationData',{XVal,YVal},...
            'ValidationFrequency',25,...
            'ValidationPatience',20,...
            'GradientThreshold',1, ...
            'LearnRateSchedule','piecewise',...
            'InitialLearnRate',0.001, ...
            'Verbose',1, ...
            'Shuffle','never');

rng('default')
N = 2;
err = zeros(N,1);
time = zeros(N,1);
Table = [];
count = 1;

for h =1 :length(numHiddenUnits)
    RMSE_Val_collect = [];
    time_taken_collect = [];
    for n = 1:N
        tic
        layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits(h),'OutputMode','sequence')
        fullyConnectedLayer(numResponses)
        regressionLayer];

        net = trainNetwork(XTrain,YTrain,layers,options);

        pred_test = predict(net,XVal);
        RMSE_val = sqrt(mean((YVal-pred_test).^2));
        

        time_taken = toc;
        
        time_taken_collect = [time_taken_collect time_taken];

        RMSE_Val_collect = [RMSE_Val_collect RMSE_val];
    end
    meanRMSEVal = mean(RMSE_Val_collect);

    meanTime = mean(time_taken_collect);

    Table(count,1) = numHiddenUnits(h);
    Table(count,2) = meanRMSEVal;
    Table(count,3) = meanTime;
    count = count + 1;
    
end
%% Find Optimal Hidden units
hiddenUnit = Table(:,1);

Rmse = Table(:,2);

Time_graph = Table(:,3);

% Plot graph
subplot(2,1,1)
plot3(hiddenUnit, Rmse, Time_graph, 'LineWidth', 2);
xlabel('Hidden Units');
ylabel('RMSE');
zlabel('Elapsed Time (seconds)');
title('Hidden Units Optimisation');
grid on

subplot(2,1,2)
plot(hiddenUnit,Time_graph,'LineWidth',2);
xlabel('Delay');
ylabel('Elapsed Time (seconds)');
title('Hidden Units Optimisation');
grid on


% Choose the optimal value 

sortTable_hu = table(hiddenUnit,Rmse,Time_graph);
sortTable_hu = sortrows(sortTable_hu,'Rmse','ascend');

optimal_hu= sortTable_hu(1, 1);
optimal_hu = table2array(optimal_hu);
 
disp("The optimal Hidden Units are:")
disp(optimal_hu)


%% Learning Rate optimisation

miniBatchSize = 10;

learning_rate = [0.5 0.1 0.01 0.001 0.0001];

rng('default')
N = 2;
err = zeros(N,1);
time = zeros(N,1);
Table_lr = [];
count = 1;

for l =1 :length(learning_rate)
    RMSE_Val_collect_lr = [];
    time_taken_collect_lr = [];
    for n = 1:N
        
        
        options = trainingOptions('adam', ...
                'MaxEpochs',optimal_ep, ...
                'MiniBatchSize',miniBatchSize, ...
                'ValidationData',{XVal,YVal},...
                'ValidationFrequency',25,...
                'ValidationPatience',20,...
                'GradientThreshold',1, ...
                'LearnRateSchedule','piecewise',...
                'InitialLearnRate',learning_rate(l), ...
                'Verbose',1, ...
                'Shuffle','never');
        
        
        tic
        layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(optimal_hu,'OutputMode','sequence')
        fullyConnectedLayer(numResponses)
        regressionLayer];

        net = trainNetwork(XTrain,YTrain,layers,options);

        pred_test = predict(net,XVal);
        RMSE_val = sqrt(mean((YVal-pred_test).^2));
        
        time_taken = toc;
        
        time_taken_collect_lr = [time_taken_collect_lr time_taken];

        RMSE_Val_collect_lr = [RMSE_Val_collect_lr RMSE_val];
    end
    meanRMSEVal = mean(RMSE_Val_collect_lr);

    meanTime = mean(time_taken_collect_lr);

    Table_lr(count,1) = learning_rate(l);
    Table_lr(count,2) = meanRMSEVal;
    Table_lr(count,3) = meanTime;
    count = count + 1;
    
end

%% Find Optimal Learning Rate
learningrate = Table_lr(:,1);

Rmse_lr = Table_lr(:,2);

Time_graph_lr = Table_lr(:,3);

subplot(2,1,1)
plot3(learningrate, Rmse_lr, Time_graph_lr, 'LineWidth', 2);
xlabel('Learning Rate');
ylabel('RMSE');
zlabel('Elapsed Time (seconds)');
title('Learning Rate Optimisation');
grid on

subplot(2,1,2)
plot(learningrate,Time_graph_lr,'LineWidth',2);
xlabel('Delay');
ylabel('Elapsed Time (seconds)');
title('Learning Rate Optimisation');
grid on

% Choose the optimal value 

sortTable_lr = table(learningrate,Rmse_lr,Time_graph_lr);
sortTable_lr = sortrows(sortTable_lr,'Rmse_lr','ascend');

optimal_lr= sortTable_lr(1, 1);
optimal_lr = table2array(optimal_lr);
 
disp("Optimal Learning Rate is:")
disp(optimal_lr)

%% Test network with optimal HU and LR and epochs

tic 
numFeatures = 6;
numResponses = 1;
miniBatchSize = 10;


    layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(optimal_hu,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
            'MaxEpochs',optimal_ep, ...
            'MiniBatchSize',miniBatchSize, ...
            'ValidationData',{XVal,YVal},...
            'ValidationFrequency',25,...
            'ValidationPatience',20,...
            'GradientThreshold',1, ...
            'LearnRateSchedule','piecewise',...
            'InitialLearnRate',optimal_lr, ...
            'Verbose',1, ...
            'Shuffle','never');

[net,info] = trainNetwork(XTrain,YTrain,layers,options);

pred_test_base = predict(net,XVal);

rmse_optimal1 = sqrt(mean((pred_test_base-YVal).^2))

time_optimal1 = toc;

disp("Optimised 1 LSTM RMSE, the result is:")
disp(rmse_optimal1)

disp("Optimised 1 LSTM time taken, the result is:")
disp(time_optimal1)


%% Adding another connectedlayer with a drop out layer to see if rmse improves

connectedLayer = [1 5 10 30 50];
dropoutPercent = [0.1 0.2 0.3 0.5 0.9];

miniBatchSize = 10;

rng('default')
N = 2;
err = zeros(N,1);
time = zeros(N,1);
Table_cd = [];
count = 1;

for c =1 :length(connectedLayer)
    for d = 1:length(dropoutPercent)
        RMSE_Val_collect_cd = [];
        time_taken_collect_cd = [];
        for n = 1:N
        
        
            options = trainingOptions('adam', ...
                    'MaxEpochs',optimal_ep, ...
                    'MiniBatchSize',miniBatchSize, ...
                    'ValidationData',{XVal,YVal},...
                    'ValidationFrequency',25,...
                    'ValidationPatience',20,...
                    'GradientThreshold',1, ...
                    'LearnRateSchedule','piecewise',...
                    'InitialLearnRate',optimal_lr, ...
                    'Verbose',1, ...
                    'Shuffle','never');


            tic
            layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(optimal_hu,'OutputMode','sequence')
            fullyConnectedLayer(connectedLayer(c))
            dropoutLayer(dropoutPercent(d))
            fullyConnectedLayer(numResponses)
            regressionLayer];

            net = trainNetwork(XTrain,YTrain,layers,options);

            pred_test = predict(net,XVal);
            RMSE_val = sqrt(mean((YVal-pred_test).^2));


            time_taken = toc;

            time_taken_collect_cd = [time_taken_collect_cd time_taken];

            RMSE_Val_collect_cd = [RMSE_Val_collect_cd RMSE_val];
        end
        meanRMSEVal = mean(RMSE_Val_collect_cd);

        meanTime = mean(time_taken_collect_cd);

        Table_cd(count,1) = connectedLayer(c);
        Table_cd(count,2) = dropoutPercent(d);
        Table_cd(count,3) = meanRMSEVal;
        Table_cd(count,4) = meanTime;
        count = count + 1;
    end 
end

%% Find Optimal Fully Connected Layer and Dropout Layer

connectedlayersize = Table_cd(:,1);

dropoutlayerpercent = Table_cd(:,2);

Rmse_cd = Table_cd(:,3);

Time_graph_cd = Table_cd(:,4);

% Choose the optimal value 

sortTable_cd = table(connectedlayersize,dropoutlayerpercent,Rmse_cd,Time_graph_cd);
sortTable_cd = sortrows(sortTable_cd,'Rmse_cd','ascend');

optimal_fc= sortTable_cd(1, 1);
optimal_fc = table2array(optimal_fc);

optimal_do = sortTable_cd(1,2);
optimal_do = table2array(optimal_do);
 
disp("Optimal Input Delay is:")
disp(optimal_fc)

disp("Optimal Feedback Delay is:")
disp(optimal_do)


%% Bayesian optimisation

% List Hyperparameters and ranges to optimise
optimVars = [
    optimizableVariable('bayes_hiddenunit',[5 400],'Type','integer')
    optimizableVariable('bayes_fullyconnected',[1 50],'Type','integer')
    optimizableVariable('bayes_dropout',[0.1 0.9], 'Transform','log')
    optimizableVariable('bayes_learnrate',[1e-3 1],'Transform','log')];

%% optimisation using bayesian 

% define layer network 
% Use Function LSTM_BayesOpt.m

newfn = @(L)LSTM_BayesOpt(XTrain,YTrain,XVal,YVal,L.bayes_learnrate,L.bayes_hiddenunit,L.bayes_fullyconnected,L.bayes_dropout);
results = bayesopt(newfn,optimVars,...
    'MaxObjectiveEvaluations', 100, ...
    'UseParallel',true);
L = bestPoint(results);

%% Plots for Bayesian Optimisation
plot(results,@plotConstraintModels,@plotMinObjective)

%% Final Model using Grid Search

tic 
numFeatures = 6;
numResponses = 1;
miniBatchSize = 10;


    layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(optimal_hu,'OutputMode','sequence')
    fullyConnectedLayer(optimal_fc)
    dropoutLayer(optimal_do)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
            'MaxEpochs',optimal_ep, ...
            'MiniBatchSize',miniBatchSize, ...
            'ValidationData',{XVal,YVal},...
            'ValidationFrequency',25,...
            'ValidationPatience',20,...
            'GradientThreshold',1, ...
            'LearnRateSchedule','piecewise',...
            'InitialLearnRate',optimal_lr, ...
            'Verbose',1, ...
            'Shuffle','never');

[net,info] = trainNetwork(XTrain,YTrain,layers,options);

pred_test_final = predict(net,YTest);

rmse_final = sqrt(mean((pred_test_final-YTest).^2))

time_final = toc;

disp("Optimised 2 LSTM RMSE, the result is:")
disp(rmse_final)

disp("Optimised 2 LSTM time taken, the result is:")
disp(time_final)

%% Final Graph for grid search model

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(pred_test_final,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Standardized Open Price")
title("Model Output againt Ground Truth for LSTM")

subplot(2,1,2)
stem(pred_test_final - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse_final)

%% LSTM model using Bayesian optimisation - FINAL MODEL

tic 
numFeatures = 6;
numResponses = 1;
miniBatchSize = 10;


    layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(L.bayes_hiddenunit,'OutputMode','sequence')
    fullyConnectedLayer(L.bayes_fullyconnected)
    dropoutLayer(L.bayes_dropout)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
            'MaxEpochs',optimal_ep, ...
            'MiniBatchSize',miniBatchSize, ...
            'ValidationData',{XVal,YVal},...
            'ValidationFrequency',25,...
            'ValidationPatience',20,...
            'GradientThreshold',1, ...
            'LearnRateSchedule','piecewise',...
            'InitialLearnRate',L.bayes_learnrate, ...
            'Verbose',1, ...
            'Shuffle','never');

[net_bayes,info] = trainNetwork(XTrain,YTrain,layers,options);

pred_test_final_bayes = predict(net_bayes,YTest);

rmse_final_bayes = sqrt(mean((pred_test_final_bayes-YTest).^2))

time_final_bayes = toc;

disp("Optimised 2 LSTM RMSE is:")
disp(rmse_final_bayes)

disp("Optimised 2 LSTM time taken  is:")
disp(time_final_bayes)

%% Final Graph

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(pred_test_final_bayes,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Standardized Open Price")
title("Model Output againt Ground Truth for LSTM")

subplot(2,1,2)
stem(pred_test_final_bayes - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse_final_bayes)


%% SAVE LSTM Final

save("BEST_LSTM_Model","net_bayes");


%% NARX Model 

%use function inputNarx to read in data
[XTrain_narx,YTrain_narx,XTest_narx, YTest_narx] = inputNarx()


%% Inputs
% reformat data for NARX model
X = tonndata(XTrain_narx,true,false);
T = tonndata(YTrain_narx,true,false);

%% BaseLine Model

inputDelay = 2;
feedbackDelay = 2;
hiddenLayerSize = 10;
trainFcn = 'trainlm';

% Define network
net = narxnet(1:inputDelay,1:feedbackDelay,hiddenLayerSize,'open',trainFcn);


%input and output processing functions
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};
% Model Inputs
net.performFcn = 'mse'

% Stopping Criteria 
net.trainParam.epochs = 1000; 

[x,xi,ai,t] = preparets(net,X,{},T);

net.divideFcn = 'divideblock';  % Divide data randomly
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0; %keep test ratio at 0 


% Train Model
tic
[net,tr] = train(net,x,t,xi,ai,'useParallel','yes');
narx_base_time = toc;

% close the network
netc = closeloop(net);
[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc);
narx_rmse_intial = sqrt(closedLoopPerformance)

disp("Initial RMSE NARX is:")
disp(narx_rmse_intial)

disp("Initial time taken NARX is:")
disp(narx_base_time)

%% hyper parameter optimisation
trials = 3; % repeat each experiment 3 times

%% Training Function

trainFcn = ["trainlm", "trainbr", "traingd"];

Narx_Table = [];
narx_count = 1;

for c = 1:length(trainFcn)
    narx_rmse_collect = [];
    narx_time_collect = [];

    for t = 1:trials
        tic
        % Define network
        net = narxnet(1:inputDelay,1:feedbackDelay,hiddenLayerSize,'open',trainFcn(c));

        [x,xi,ai,t] = preparets(net,X,{},T);

        net.divideFcn = 'divideblock';  % Divide data 
        net.divideMode = 'time';  % Divide up every sample
        net.divideParam.trainRatio = 80/100;
        net.divideParam.valRatio = 20/100;
        net.divideParam.testRatio = 0;


        %input and output processing functions
        net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

        % Model Inputs
        net.performFcn = 'mse'

        % Stopping Criteria 
        net.trainParam.epochs = 1000; 
        net.trainParam.max_fail = 20;

        % Train Model
        [net,tr] = train(net,x,t,xi,ai,'useParallel','yes');

        % close the network
        netc = closeloop(net);

        [xc,xic,aic,tc] = preparets(netc,X,{},T);
        yc = netc(xc,xic,aic);
        ec = gsubtract(tc,yc);
        errorc = cell2mat(ec);
        narx_rmse_optimise = mean(sqrt(mean(errorc.^2)));

        narx_time_taken = toc;

        narx_rmse_collect = [narx_rmse_collect narx_rmse_optimise];

        narx_time_collect = [narx_time_collect narx_time_taken];

        end
        meanNarxRMSE = mean(narx_rmse_collect);
        meanNarxTime = mean(narx_time_collect);

        Narx_Table{narx_count,1} = char(trainFcn(c));
        Narx_Table{narx_count,2} = meanNarxRMSE;
        Narx_Table{narx_count,3} = meanNarxTime;
        narx_count = narx_count + 1;
end

%% Optimal training function
TrainingFunction = Narx_Table(:,1);

Rmse_tf = Narx_Table(:,2);

Time_graph_tf = Narx_Table(:,3);

% Choose the optimal value 

sortTable_tf = table(TrainingFunction,Rmse_tf,Time_graph_tf);
sortTable_tf = sortrows(sortTable_tf,'Rmse_tf','ascend');

optimal_tf= sortTable_tf(1, 1);
optimal_tf = table2array(optimal_tf);

disp("Optimal training function, the result is:")
disp(optimal_tf)



%% Hidden Layer size optimisation

hiddenLayerSize = [5 10 25 50 75 100];

Narx_Table_hl = [];
narx_count_hl = 1;

    for h = 1:length(hiddenLayerSize)
        narx_rmse_collect_hl = [];
        narx_time_collect_hl = [];

        for t = 1:trials
        tic
        % Define network
        net = narxnet(1:inputDelay,1:feedbackDelay,hiddenLayerSize(h),'open',char(optimal_tf));

        [x,xi,ai,t] = preparets(net,X,{},T);

        net.divideFcn = 'divideblock';  % Divide data 
        net.divideMode = 'time';  % Divide up every sample
        net.divideParam.trainRatio = 80/100;
        net.divideParam.valRatio = 20/100;
        net.divideParam.testRatio = 0;


        %input and output processing functions
        net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

        % Model Inputs
        net.performFcn = 'mse'

        % Stopping Criteria 
        net.trainParam.epochs = 1000; 
        net.trainParam.max_fail = 20;

        % Train Model
        [net,tr] = train(net,x,t,xi,ai,'useParallel','yes');

        % close the network
        netc = closeloop(net);

        [xc,xic,aic,tc] = preparets(netc,X,{},T);
        yc = netc(xc,xic,aic);
        closedLoopPerformance = perform(net,tc,yc);
        narx_rmse_optimise_hl = sqrt(closedLoopPerformance);

        narx_time_taken_hl = toc;

        narx_rmse_collect_hl = [narx_rmse_collect_hl narx_rmse_optimise_hl];

        narx_time_collect_hl = [narx_time_collect_hl narx_time_taken_hl];

    end
    meanNarxRMSE_hl = mean(narx_rmse_collect_hl);
    meanNarxTime_hl = mean(narx_time_collect_hl);

    Narx_Table_hl(narx_count_hl,1) = hiddenLayerSize(h);
    Narx_Table_hl(narx_count_hl,2) = meanNarxRMSE_hl;
    Narx_Table_hl(narx_count_hl,3) = meanNarxTime_hl;
    narx_count_hl = narx_count_hl + 1;
end


%% Optimal hidden layer
HiddenLayer = Narx_Table_hl(:,1);

Rmse_hl = Narx_Table_hl(:,2);

Time_graph_hl = Narx_Table_hl(:,3);

subplot(2,1,1);
plot3(HiddenLayer, Rmse_hl, Time_graph_hl, 'LineWidth', 2);
xlabel('Hidden Layer Size');
ylabel('RMSE');
zlabel('Elapsed Time (seconds)');
title('Hidden Layer Size Optimisation');
grid on

subplot(2,1,2)
plot(HiddenLayer,Time_graph_hl,'LineWidth',2);
xlabel('Hidden Layer Size');
ylabel('Elapsed Time (seconds)');
title('Hidden Layer Size Optimisation');
grid on

% Choose the optimal value 

sortTable_hl = table(HiddenLayer,Rmse_hl,Time_graph_hl);
sortTable_hl = sortrows(sortTable_hl,'Rmse_hl','ascend');

optimal_hl= sortTable_hl(1, 1);
optimal_hl = table2array(optimal_hl);
 
disp("Optimal hidden layer size is:")
disp(optimal_hl)

%%  delay optimisation

% keep input delay and feedback delay same value

Delay = [2 5 10 20 25 50];

Narx_Table_d = [];
narx_count_d = 1;

for d = 1:length(Delay)
    narx_rmse_collect_d = [];
    narx_time_collect_d = [];
    for t = 1:trials
        tic
        % Define network
        net = narxnet(1:Delay(d),1:Delay(d),optimal_hl,'open',char(optimal_tf));

        [x,xi,ai,t] = preparets(net,X,{},T);

        net.divideFcn = 'divideblock';  % Divide data 
        net.divideMode = 'time';  % Divide up every sample
        net.divideParam.trainRatio = 80/100;
        net.divideParam.valRatio = 20/100;
        net.divideParam.testRatio = 0;


        %input and output processing functions
        net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

        % Model Inputs
        net.performFcn = 'mse'

        % Stopping Criteria 
        net.trainParam.epochs = 1000; 
        net.trainParam.max_fail = 20;

        % Train Model
        [net,tr] = train(net,x,t,xi,ai,'useParallel','yes');

        % close the network
        netc = closeloop(net);

        [xc,xic,aic,tc] = preparets(netc,X,{},T);
        yc = netc(xc,xic,aic);
        closedLoopPerformance = perform(net,tc,yc);
        narx_rmse_optimise_d = sqrt(closedLoopPerformance);

        narx_time_taken_d = toc;

        narx_rmse_collect_d = [narx_rmse_collect_d narx_rmse_optimise_d];

        narx_time_collect_d = [narx_time_collect_d narx_time_taken_d];

    end
    meanNarxRMSE_d = mean(narx_rmse_collect_d);
    meanNarxTime_d = mean(narx_time_collect_d);

    Narx_Table_d(narx_count_d,1) = Delay(d);
    Narx_Table_d(narx_count_d,2) = meanNarxRMSE_d;
    Narx_Table_d(narx_count_d,3) = meanNarxTime_d;
    narx_count_d = narx_count_d + 1;
end

%% Optimal Delay
Delay = Narx_Table_d(:,1);

Rmse_d = Narx_Table_d(:,2);

Time_graph_d = Narx_Table_d(:,3);

subplot(2,1,1)
plot3(Delay, Rmse_d, Time_graph_d, 'LineWidth', 2);
xlabel('Delay');
ylabel('RMSE');
zlabel('Elapsed Time (seconds)');
title('Delay Optimisation');
grid on

subplot(2,1,2)
plot(Delay,Time_graph_d,'LineWidth',2);
xlabel('Delay');
ylabel('Elapsed Time (seconds)');
title('Delay Optimisation');
grid on

% Choose the optimal value 

sortTable_d = table(Delay,Rmse_d,Time_graph_d);
sortTable_d = sortrows(sortTable_d,'Rmse_d','ascend');

optimal_d= sortTable_d(1, 1);
optimal_d = table2array(optimal_d);
 
disp("Optimal Delay is:")
disp(optimal_d)


%% Vary input delay and feedback delay 

inputDelay =  [2 5 20]; %[2 4 5 10 20 25 30];
feedbackDelay = [2 5 10]; %[2 4 5 10 20 25 30];

Narx_Table_fd = [];
narx_count_fd = 1;

for i = 1:length(inputDelay)
    for f=1:length(feedbackDelay)
        narx_rmse_collect_fd = [];
        narx_time_collect_fd = [];
        for t = 1:trials
            tic
            % Define network
            net = narxnet(1:inputDelay(i),1:feedbackDelay(f),optimal_hl,'open',char(optimal_tf));

            [x,xi,ai,t] = preparets(net,X,{},T);

            net.divideFcn = 'divideblock';  % Divide data 
            net.divideMode = 'time';  % Divide up every sample
            net.divideParam.trainRatio = 80/100;
            net.divideParam.valRatio = 20/100;
            net.divideParam.testRatio = 0;


            %input and output processing functions
            net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
            net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

            % Model Inputs
            net.performFcn = 'mse'

            % Stopping Criteria 
            net.trainParam.epochs = 1000; 
            net.trainParam.max_fail = 20;

            % Train Model
            [net,tr] = train(net,x,t,xi,ai,'useParallel','yes');

            % close the network
            netc = closeloop(net);

            [xc,xic,aic,tc] = preparets(netc,X,{},T);
            yc = netc(xc,xic,aic);
            closedLoopPerformance = perform(net,tc,yc);
            narx_rmse_optimise_fd = sqrt(closedLoopPerformance);

            narx_time_taken_fd = toc;

            narx_rmse_collect_fd = [narx_rmse_collect_fd narx_rmse_optimise_fd];

            narx_time_collect_fd = [narx_time_collect_fd narx_time_taken_fd];

        end
        meanNarxRMSE_fd = mean(narx_rmse_collect_fd);
        meanNarxTime_fd = mean(narx_time_collect_fd);

        Narx_Table_fd(narx_count_fd,1) = inputDelay(i);
        Narx_Table_fd(narx_count_fd,2) = feedbackDelay(f);
        Narx_Table_fd(narx_count_fd,3) = meanNarxRMSE_fd;
        Narx_Table_fd(narx_count_fd,4) = meanNarxTime_fd;
        narx_count_fd = narx_count_fd + 1;
    end
end


%% Optimal input Delay and feedback delay

inputDelay_op = Narx_Table_fd(:,1);

feedbackDelay_op = Narx_Table_fd(:,2);

Rmse_op = Narx_Table_fd(:,3);

Time_graph_op = Narx_Table_fd(:,4);

% plot3(Delay, Rmse_d, Time_graph_d, 'LineWidth', 2);
% xlabel('Delay');
% ylabel('RMSE');
% zlabel('Elapsed Time (seconds)');
% title('Delay Optimisation');
% grid on

% Choose the optimal value 

sortTable_op = table(inputDelay_op,feedbackDelay_op,Rmse_op,Time_graph_op);
sortTable_op = sortrows(sortTable_op,{'Rmse_op' 'Time_graph_op'},'ascend');

optimal_input= sortTable_op(1, 1);
optimal_input = table2array(optimal_input);

optimal_feedback = sortTable_op(1,2);
optimal_feedback = table2array(optimal_feedback);
 
disp("Optimal Input Delay is:")
disp(optimal_input)

disp("Optimal Feedback Delay is:")
disp(optimal_feedback)

%% Bayesian Optimisation

%% hyperparameters 

optimVars_NARX = [
    optimizableVariable('bayes_inputdelay',[2 30],'Type','integer')
    optimizableVariable('bayes_feedbackdelay',[2 30],'Type','integer')
    optimizableVariable('bayes_hiddenlayer',[5 200], 'Type','integer')
    optimizableVariable('bayes_training',{'trainlm','trainbr','traingd'},'Type','categorical')];

%% optimisation using bayesian 

% define layer network

newfn_narx = @(N)NARX_BayesOpt(X,T,N.bayes_inputdelay,N.bayes_feedbackdelay,N.bayes_hiddenlayer,N.bayes_training);
results_narx = bayesopt(newfn_narx,optimVars_NARX,...
    'MaxObjectiveEvaluations', 100, ...
    'UseParallel',true,...
    'PlotFcn',{@plotObjectiveModel,@plotMinObjective,@plotElapsedTime});
N = bestPoint(results_narx);


%% Test model using Bayesian Optimisation


%% Final Narx Model

%Train optimal network

% Define network
net = narxnet(1:N.bayes_inputdelay,1:N.bayes_feedbackdelay,N.bayes_hiddenlayer,'open',char(N.bayes_training));

[x,xi,ai,t] = preparets(net,X,{},T);

net.divideFcn = 'divideblock';  % Divide data 
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0;


%input and output processing functions
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Model Inputs
net.performFcn = 'mse'

% Stopping Criteria 
net.trainParam.epochs = 1000; 
net.trainParam.max_fail = 20;

% Train Model
[net,tr] = train(net,x,t,xi,ai,'useParallel','yes');

% close the network
netc = closeloop(net);

[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc);
narx_rmse_final_test = sqrt(closedLoopPerformance);

disp("Final RMSE NARX, the result is:")
disp(narx_rmse_final_test)

XT_1 = tonndata(XTest_narx,true,false);
TT_1 = tonndata(YTest_narx,true,false);

[xt,xit,ait,tt] = preparets(netc,XT_1,{},TT_1);
yt = netc(xt,xit,ait);
ect = gsubtract(tt,yt);
errorct = cell2mat(ect);
Rmse_Test_second = sqrt(mean(errorct.^2));

disp("Final RMSE NARX Test, the result is:")
disp(Rmse_Test_second)

%%

% Model vs Truth graph 
TS_1 = size(tt,2);
plot(1:TS_1,cell2mat(tt),'b',1:TS_1,cell2mat(yt),'r')
title('Model Output against Ground Truth for NARX')
xlabel('Month')
ylabel('Standardised Open Price')
legend(["Observed" "Predicted"])

%% Final Narx Model

%Train optimal network
tic
% Define network
net = narxnet(1:optimal_input,1:optimal_feedback,optimal_hl,'open',char(optimal_tf));

[x,xi,ai,t] = preparets(net,X,{},T);

net.divideFcn = 'divideblock';  % Divide data 
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0;


%input and output processing functions
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Model Inputs
net.performFcn = 'mse'

% Stopping Criteria 
net.trainParam.epochs = 1000; 
net.trainParam.max_fail = 20;

% Train Model
[net,tr] = train(net,x,t,xi,ai,'useParallel','yes');

% close the network
netc = closeloop(net);

[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc);
narx_rmse_final = sqrt(closedLoopPerformance);

disp("Final RMSE NARX, the result is:")
disp(narx_rmse_final)

view(net)
view(netc)

% test network 

XT = tonndata(XTest_narx,true,false);
TT = tonndata(YTest_narx,true,false);

[xt,xit,ait,tt] = preparets(netc,XT,{},TT);
yt = netc(xt,xit,ait);
ect = gsubtract(tt,yt);
errorct = cell2mat(ect);
Rmse_Test = sqrt(mean(errorct.^2));

final_narx_time = toc;

disp("Final RMSE NARX Test, is:")
disp(Rmse_Test)

disp("Final Time NARX Test, is:")
disp(final_narx_time)
%% Final Graph

% Model vs Truth graph 
TS = size(tt,2);
plot(1:TS,cell2mat(tt),'b',1:TS,cell2mat(yt),'r')
title('Model Output against Ground Truth for NARX')
xlabel('Month')
ylabel('Standardised Open Price')
legend(["Observed" "Predicted"])

%% Save Narx Model

save("BEST_NARX_Model","net");  

%% Save Test data as csv 

writematrix(XTest,'TestData.csv');