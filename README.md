# EMA_LSTM_model

## Description
LSTM (Long Short-Term Memory) is a type of neural network designed for sequential data like time series, speech, and text. It's capable of selectively retaining and forgetting information from past inputs, allowing it to learn patterns and make predictions over long time horizons. The following LSTM has been tuned to predict whether or not a participant of the EMA study will be suicidal on a given day. 

## Model Process

## Config
'''
Data: Path to CSV to train on.
Features: List of features used by the model. Adjust this file to experiment with different combinations of features to judge accuracy.
Window_Size: Number of days the sliding window uses to gather data for its predictions. Originally set and tested at 5 days.
Validation_Loop: Number of times the validation loop randomly assigns the participants as training and validation and reruns the model. 

## Dependencies
make sure to address this somewhere (it would be best to package it somehow)

## Features
There are some limitations on what can be included and what should not be excluded using the config. I plan to try to clear up as many discrepancies here as possible.

## Output
1. Single Box Plot for the run
2. Statistics for the run saved in a CSV
