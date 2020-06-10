%% LSTM and NARX Testing

%% Load Test Data

Test_Data = readtable('TestData.xlsx');

%% Load LSTM Final Model 

load('BEST_LSTM_Model.mat')

%% Format test data for LSTM model

Test_Data = Test_Data{:,:};
XTest_Final = Test_Data.';
YTest_Final = Test_Data(:,1).';

%% Predict LSTM 

predictFinalLSTM = predict(net_bayes,XTest_Final);

RMSEFinalLstm = sqrt(mean((predictFinalLSTM-YTest_Final).^2))

figure
subplot(2,1,1)
plot(YTest_Final)
hold on
plot(predictFinalLSTM,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Standardized Open Price")
title("Model Output againt Ground Truth for LSTM")

subplot(2,1,2)
stem(predictFinalLSTM - YTest_Final)
xlabel("Month")
ylabel("Error")
title("RMSE = " + RMSEFinalLstm)

%% Load NARX Final Model

load('BEST_NARX_Model.mat')

%% Prepare Data For NARX Model

XT_final = tonndata(XTest_Final,true,false);
TT_final = tonndata(YTest_Final,true,false);

%% Predict Final NARX Model

% close the network for predictions
netc = closeloop(net);

[xt_final,xit_final,ait_final,tt_final] = preparets(netc,XT_final,{},TT_final);
yt_final = netc(xt_final,xit_final,ait_final);
ect_final = gsubtract(tt_final,yt_final);
errorct_final = cell2mat(ect_final);
RMSENarxFinal = sqrt(mean(errorct_final.^2));

%Display the RMSE Score on test set
disp("Final RMSE NARX Test, is:")
disp(RMSENarxFinal)

% Model vs Truth graph 
TS_final = size(tt_final,2);
plot(1:TS_final,cell2mat(tt_final),'b',1:TS_final,cell2mat(yt_final),'r')
title('Model Output against Ground Truth for NARX')
xlabel('Month')
ylabel('Standardised Open Price')
legend(["Observed" "Predicted"])