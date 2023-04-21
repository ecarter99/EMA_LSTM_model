# EMA_LSTM_model

## Setup
1. Setup Anaconda https://docs.anaconda.com/free/anaconda/install/
2. Setup Git on your system https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
3. Navigate to the Directory where you want to run the models in your terminal/commandline
4. Clone the repository using the following command:
```
git clone https://github.com/ecarter99/EMA_LSTM_model.git
```
5. Setup the environment using the following command:
```
conda create --name LSTM_ENV --file environment.txt
```

## Running Model
```
python LSTM_MODEL.py
```

## Description
LSTM (Long Short-Term Memory) is a type of neural network designed for sequential data like time series, speech, and text. It's capable of selectively retaining and forgetting information from past inputs, allowing it to learn patterns and make predictions over long time horizons. The following LSTM has been tuned to predict whether or not a participant of the EMA study will be suicidal on a given day. 

## Model Process

## Config
'''
Data: Path to CSV to train on.
Features: List of features used by the model. Adjust this file to experiment with different combinations of features to judge accuracy.
Window_Size: Number of days the sliding window uses to gather data for its predictions. Originally set and tested at 5 days.
Validation_Loop: Number of times the validation loop randomly assigns the participants as training and validation and reruns the model. 
'''
## Dependencies
1. Anaconda
2. Git
3. A full list of dependencies can be found in the environment.txt file used to setup the conda environment

## Features
There are some limitations on what can be included and what should not be excluded using the config. I plan to try to clear up as many discrepancies here as possible.

## Output
All output can be found in the output/ directory. Each run is specified by the timestamp at which the model was executed. 

1. Single Box Plot for the run
2. Statistics for the run saved in a CSV

