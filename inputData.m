function [raw_data, train_T, train_I, test_T, test_I, validation_T, validation_I] = inputData()

% Get Inputs
% Load data for LSTM Model

data = readtable('SP500_standardise.csv');
data = table2timetable(data);

%Partition the Training data manually
dataTrain = data(1:5112, :);
dataValidation = data(5113:6208, :);
dataTest = data(6209:end, :);

%use first column as the target (Open price)

train_T = dataTrain{:,1};
%Use all columns as inputs
train_I = dataTrain(:, 1:end);
train_I = timetable2table(train_I);
train_I = table2array(train_I(:, 2:end));

validation_T = dataValidation{:,1};
validation_I = dataValidation(:, 1:end);
validation_I = timetable2table(validation_I);
validation_I = table2array(validation_I(:, 2:end));

test_T = dataTest{:,1};
test_I = dataTest(:, 1:end);
test_I = timetable2table(test_I);
test_I = table2array(test_I(:, 2:end));

raw_data = data;

% Transpose Data to use as input for LSTM model
train_I = train_I.';
train_T = train_T.';

test_T = test_T.';
test_I = test_I.';

validation_T = validation_T.';
validation_I = validation_I.';

end

