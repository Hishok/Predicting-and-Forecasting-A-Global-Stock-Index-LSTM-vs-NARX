function [XTrain_narx,YTrain_narx,XTest_narx, YTest_narx] = inputNarx()

% function to create data for NARX model

% read the standardised data 
narx_data = readtable('SP500_standardise.csv');
narx_data = table2timetable(narx_data);

%manually split the data 
narx_dataTrain = narx_data(1:6208, :);
narx_dataTest = narx_data(6209:end, :);

% Obtain XTrain and YTrain
YTrain_narx = narx_dataTrain{:,1};
XTrain_narx = narx_dataTrain(:, 1:end);
XTrain_narx = timetable2table(XTrain_narx);
XTrain_narx = table2array(XTrain_narx(:, 2:end));

% Obtain XTest and YTest
YTest_narx = narx_dataTest{:,1};
XTest_narx = narx_dataTest(:, 1:end);
XTest_narx = timetable2table(XTest_narx);
XTest_narx = table2array(XTest_narx(:, 2:end));

%Transpose the data
YTest_narx = YTest_narx.';
XTest_narx = XTest_narx.';

XTrain_narx = XTrain_narx.';
YTrain_narx = YTrain_narx.';

narx_raw_data = narx_data

end

