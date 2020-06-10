###################################
Predicting and Forecasting A Global Stock Index: LSTM vs NARX
INM427 Neural Computing Coursework
Hisho Rajanathan
###################################

Files in Folder: 

1. Coursework Report: INM427 Report.pdf

2. Test Data - TestData.xlsx

3. LSTM_NARX.m - contains base model, optimisation steps and final model for LSTM and NARX

4. inputData.m - function that loads the train, validation and test data for LSTM

5. inputNarx.m - function that loads the train and test data for NARX

6. LSTM_BayesOpt.m - function for the LSTM model to perform Bayesian Optimisation

7. NARX_BayesOpt.m - function for the NARX model to perform Bayesian Optimisation

8. FinalBestModels.m - test the final optimised models on test data

9. BEST_LSTM_Model.mat - final optimised LSTM model

10. BEST_NARX_Model.mat - final optimised NARX model

11. SP500.csv - original dataset

12. SP500_standardise.csv - original dataset standardised

13. Neural Computing Coursework.ipynb  - data collection and preprocessing

Open functions 1 – 4 in order to run script 5. Script 6 allows to test the final optimised models using 7 & 8 which contains the final optimised models for LSTM and NARX.