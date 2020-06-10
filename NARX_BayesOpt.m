function [RMSE_ave_narx] = NARX_BayesOpt(X,T,bayes_inputdelay,bayes_feedbackdelay,bayes_hiddenlayer,bayes_training)


% Define network
net = narxnet(1:bayes_inputdelay,1:bayes_feedbackdelay,bayes_hiddenlayer,'open',char(bayes_training));

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
net.performFcn = 'mse' %Performance indicator is MSE

% Stopping Criteria 
net.trainParam.epochs = 1000; 
net.trainParam.max_fail = 20;

% Train Model
[net,tr] = train(net,x,t,xi,ai,'useParallel','yes');

% close the network for prediction
netc = closeloop(net);

[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc);
RMSE_narx = sqrt(closedLoopPerformance);

RMSE_ave_narx = mean(RMSE_narx);
end

