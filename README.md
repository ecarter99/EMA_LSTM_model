# EMA_LSTM_model

## Setup
1. Setup Anaconda https://docs.anaconda.com/free/anaconda/install/
2. Setup Git on your system https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
3. Navigate to the Directory where you want to run the models in your terminal/commandline
4. Clone the repository using the following command:
```
git clone https://github.com/ecarter99/EMA_LSTM_model.git
cd EMA_LSTM_model
```
5. Setup the environment using the following command:
```
conda create --name LSTM_ENV --file environment.txt
```
6. Activate the environment (if you exit the terminal or anything resets you'll have to activate when you come back. The command is the same each time - when activated, the terminal will show (LSTM_ENV) before the line you're working on)
```
conda activate LSTM_ENV
```

## Running Model
```
python LSTM_MODEL.py
```

## Description
LSTM (Long Short-Term Memory) is a type of neural network designed for sequential data like time series, speech, and text. It's capable of selectively retaining and forgetting information from past inputs, allowing it to learn patterns and make predictions over long time horizons. The following LSTM has been tuned to predict whether or not a participant of the EMA study will be suicidal on a given day. 

## Data Cleaning

All the data cleaning for this model is performed in the preprocess_encodings() function. The function pulls out the relevant features to use for the model and drops rows with null values. Categorical features using 'yes' or 'no' are changed to 1, 0. Participant groups are onehot encoded and finally the data is reshaped to fit the needs of the PyTorch LSTM model requirements.

Currently the model is tuned for evening surveys. Future work would be to fit it for the entire set of surveys and other data.

## Model Process
This LSTM relies on PyTorch neural-network (nn) libraries. The parameters for the neural network are as follows but can be adjusted and tuned:
```
hidden_layer_size = 128
num_layers = 1
bidirectional = True
p_dropout = 0
```
Some tuning of these would be recommended.

The default collate function was overridden to oversample suicidal days since our data has a heavy skew towards the non-suicidal target. This oversampling was implimented in each minibatch by randomly splitting the batch in half and overfitting on of the halves to suicidal and the other to suicidal days. To see specifically how this was implimented, reference the custom_collate_fn() in the code.

Validation was performed via a custom cross-validation method. Over 20 iterations (can be adjusted using the config), 3 participants are chosen at random to be witheld from model training to be used for validation. Accuracies are reported for each iteration in the results and the average accuracy is reported in a boxplot also in the output.

## Config
```
Data: Path to CSV to train on.
Features: List of features used by the model. Adjust this file to experiment with different combinations of features to judge accuracy.
Window_Size: Number of days the sliding window uses to gather data for its predictions. Originally set and tested at 5 days.
Validation_Loop: Number of times the validation loop randomly assigns the participants as training and validation and reruns the model. 
```
## Dependencies
1. Anaconda
2. Git
3. A full list of dependencies can be found in the environment.txt file used to setup the conda environment

## Features
** Features included can be found in the script in the preprocess_encodings() function ** future work would be to include the option to adjust which features to use in the config file

## Output
All output can be found in the output/ directory. Each run is specified by the timestamp at which the model was executed. 

1. Single Box Plot for the run
2. Statistics for the run saved in a CSV

## Results
Baseline accuracy for the model is 50%. This figure comes from the overfitting process used above where half of the data represented is non-suicidal and the other half is. 

From the output, we find that the accuracy averages aboud 80% (depending on your run - and also tuning to be performed). This 30% in accuracy greater than the baseline tells us that the features in the dataset have predictive power. Some runs perform higher and others lower which could be analyzed further, but it is very insightful that such an improved prediction can be achieved with this LSTM.
