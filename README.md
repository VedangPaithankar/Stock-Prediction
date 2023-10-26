# Samsung Stock Price Prediction with LSTM

This project involves building a Long Short-Term Memory (LSTM) neural network model to predict Samsung stock prices based on historical data. 

## Introduction
Stock price prediction is a challenging task in the field of finance. This project aims to predict Samsung stock prices using LSTM, a type of recurrent neural network (RNN) known for its ability to capture sequential patterns.

## Model Architecture

The core of this project is the LSTM neural network, which is a type of recurrent neural network (RNN) capable of capturing patterns in time series data. The model architecture is as follows:

1. **Data Preprocessing:** The raw dataset is preprocessed to ensure compatibility with the LSTM model. This may include normalization, feature scaling, and splitting the data into training and testing sets.

2. **LSTM Layers:** The LSTM layers are designed to take sequences of historical data as input and generate predictions. You can specify the number of LSTM layers and units in each layer based on experimentation and hyperparameter tuning.

3. **Output Layer:** The output layer is a single neuron, which produces the predicted stock price.

4. **Training:** The model is trained on the training data using a loss function and an optimizer. Training involves forward and backward passes, and it aims to minimize the prediction error.

5. **Evaluation:** The model is evaluated using the testing data to assess its performance. Common evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Analysis

### Closing Trend
![Closing Trend](/CloseTrend.png)

### 100 Days Moving Average
![100 Days Moving Average](/100DaysMovingAvg.png)

### 200 Days Moving Average
![200 Days Moving Average](/200DaysMovingAvg.png)

### Predicted Model Trend
![Predicted Model](/PredictedModelTrend.png)
