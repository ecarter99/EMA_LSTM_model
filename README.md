# EMA_LSTM_model

## Description
LSTM (Long Short-Term Memory) is a type of neural network designed for sequential data like time series, speech, and text. It's capable of selectively retaining and forgetting information from past inputs, allowing it to learn patterns and make predictions over long time horizons. The following LSTM has been tuned to predict whether or not a participant of the EMA study will be suicidal on a given day. 

## Model Process

## Config
'''
Features: List of features used by the model. Adjust this file to experiment with different combinations of features to judge accuracy.
Window_Size: Number of days the sliding window uses to gather data for its predictions. Originally set and tested at 5 days.
Validation_Loop: Number of times the validation loop randomly assigns the participants as training and validation and reruns the model. 
'''
