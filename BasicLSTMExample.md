# Basic LSTM Example (One Dimensional Time Series Forecasting)

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
```

This document offers a foundational guide for constructing a basic Long Short-Term Memory (LSTM) model. 
As a specialized form of recurrent neural network (RNN), the LSTM model excels in capturing dependencies
in sequential data, making it highly effective for various applications such as time series forecasting 
and natural language processing. This example specifically focuses on demonstrating the use of LSTM models 
for time series prediction, showcasing their significance as a pivotal tool for analyzing and forecasting 
sequential data.


The provided code snippet outlines the definition of ComplexLSTMModel, a structured and user-friendly
implementation of a Long Short-Term Memory (LSTM) network using PyTorch's neural network module (nn.Module).
At its core, the model is designed to handle sequential predictions by allowing customization across three 
key parameters: the number of features in the input (input_dim, the dimension of the time series, mostly our 
project contains only 1 dimension time series, e.g. the prediction of the temperature), the number of hidden 
units in each LSTM layer (hidden_dim, the dimension of the h at each step), and the total number of LSTM layers (num_layers). 
The architecture is initiated with a single LSTM layer, configurable for different complexities of sequence 
data processing, and is followed by a linear layer that maps the LSTM outputs to the desired output dimension.
The forward method of the model defines the flow of data through the network, with the LSTM layer processing the input 
sequence and the linear layer producing the final output. 

```python
# Define the LSTM model:
class ComplexLSTMModel(nn.Module):
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units
    # num_layers : number of LSTM layers
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ComplexLSTMModel, self).__init__()  # initializes the parent class nn.Module
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # defines forward pass of the neural network
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out
```

Let us define some basic functions to generate time series data. The function generate_time_series is to create 
synthetic time series data, which can be instrumental in testing and developing time series forecasting models. 
 
```python
def generate_time_series(num_series, length, slopes, seasonality_amplitudes, noise_levels):
    """
    Generate a fake time series data with different features for each series.

    Parameters:
    - num_series: Number of time series (features) to generate.
    - length: The length of each time series.
    - slopes: List of slopes, one per time series.
    - seasonality_amplitudes: List of seasonality amplitudes, one per time series.
    - noise_levels: List of noise levels (standard deviation), one per time series.

    Returns:
    A NumPy array of shape (length, num_series), representing the generated time series data.
    """
    time = np.arange(length)
    time_series_data = np.zeros((length, num_series))

    for i in range(num_series):
        trend = slopes[i] * time
        seasonal = seasonality_amplitudes[i] * np.sin(2 * np.pi * time / 365)
        noise = np.random.normal(0, noise_levels[i], length)
        time_series = trend + seasonal + noise
        time_series_data[:, i] = time_series

    normalized_data = (time_series_data - np.mean(time_series_data, axis=0)) / np.std(time_series_data, axis=0)

    return normalized_data
```

The following code specifies the dataset size, the structure of the input data, and the computational resources 
to be used. With 2000 samples, each represented by a single feature, the configuration is tailored for univariate 
time series forecasting (for simplicity, input_dim = 1). A window size of 50 time steps dictates how much past data 
is considered for making future predictions. The model will be trained in batches of 16 samples, utilizing a GPU if 
available. 

```python
num_samples = 2000
input_dim = 1
external_dim = 0
batch_size = 16
window = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

We will generate some time series data with different features for each series. The function takes the number of time 
series to generate, the length of each time series, and the slopes, seasonality amplitudes, and noise levels for each
time series as input. 

```python
num_series = input_dim + external_dim
slopes = np.random.uniform(0.01, 0.1, num_series)  # Random slopes between 0.01 and 0.1
seasonality_amplitudes = np.random.uniform(5, 15, num_series)  # Random seasonality amplitudes
noise_levels = np.random.uniform(1, 5, num_series)  # Random noise levels between 1 and 5
features = generate_time_series(num_series, num_samples, slopes, seasonality_amplitudes, noise_levels)
```

We are going to follow 8:2 rule to split the data into training and validation sets. We will use the first 80% of 
the data for training and the remaining 20% for validation. We will use the training_size variable to store the size 
of the training set and the val_size variable to store the size of the validation set. We will then use the 
training_size variable to split the features array into training and validation sets. I think random_split is a 
better option to split the data into training and validation sets but I am using the following code to split the
data into training and validation sets at this moment. A separate document will be created to talk about the usage of
the random_split function. 

```python
training_size = int(num_samples * 0.8)
val_size = num_samples - training_size
train_data, val_data = features[0:training_size, :], features[training_size:num_samples, :]
```

We will use the MinMaxScaler class from the scikit-learn library to scale the training and validation sets. Then the 
fit_transform method is used to fit the scaler to the training set and transform the training set. 

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.fit_transform(val_data)
```

To prepare our training data for the LSTM model, we will organize it into sequences and corresponding labels. 
The sequence_length variable will determine how many time steps we consider in each sequence, 
essentially defining the amount of historical data the model will use to make predictions. 
We will traverse the training dataset, generating input sequences and their associated labels. 
Each label sequence mirrors its corresponding input sequence but is offset by one time step forward, 
indicating the next value the model should predict.

```python
# Create sequences and labels for training data
# Here should be the work of tensor_data set class
sequence_length = 50  # Number of time steps to look back
X_train, y_train = [], []
for i in range(len(scaled_train) - sequence_length):
    X_train.append(scaled_train[i:i + sequence_length])
    y_train.append(scaled_train[i + 1:i + sequence_length + 1])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
```
We will do similar work for the validation set. 

```python
# Create sequences and labels for testing data
sequence_length = 30  # Number of time steps to look back
X_test, y_test = [], []
for i in range(len(scaled_test) - sequence_length):
    X_test.append(scaled_test[i:i + sequence_length])
    y_test.append(scaled_test[i + 1:i + sequence_length + 1])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
```

The Dataloader is a function designed to streamline the batching of data for model training.
It produces an iterable in Python, simplifying the process of retrieving data in specified batch sizes. 
This functionality is particularly useful for efficiently handling large datasets by dividing them into manageable 
batches, thereby optimizing the training process. Importantly, the input to the Dataloader is expected to be 
of the tensor dataset class, ensuring that data is appropriately formatted for processing within the training loop.

```python
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
```  
The following code snippet defines a set of parameters for an LSTM model and then initializes the model using 
these parameters.

```python
# Define model parameters
model_params = {
    "input_dim": input_dim + external_dim,
    "hidden_dim": 64,
    "num_layers": 2,
}

model = ComplexLSTMModel(**model_params)
```

To set up the model, we initialize it and specify the MSE loss function for evaluating performance, alongside the Adam 
optimizer with a learning rate of 0.001. The training will span 100 epochs. Additionally, we utilize the .to method to
allocate the model to the appropriate device, either CPU or GPU. The early stopping option will be added in other 
documents.

```python
def train_with_early_stopping(model, data_loaders, num_epochs=100,
                              window=100, num_features=1, batch_size=64, optimizer=None,
                              scheduler=None):
    """
    Train a neural network model, non early stopping version.
    """
    
    criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if scheduler is None:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # early_stopping_counter = 0
    # min_val_loss = float("inf")

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store losses for plotting
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for idx, (sequence_data, target_data) in enumerate(train_loader):
            sequence_data = sequence_data.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()

            # training loop
            out = model(sequence_data)
            loss = criterion(out, target_data)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')
        #scheduler.step()

        window = 30  # for test data
        avg_val_loss = validate(model, val_loader, criterion, batch_size, window, num_features, device)
        epoch_val_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}')
        
def validate(model, val_loader, criterion, batch_size, window, num_features, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # No gradients needed
        for idx, (sequence_data, targets) in enumerate(val_loader):
            sequence_data, targets = sequence_data.to(device), targets.to(device)
            outputs = model(sequence_data.reshape(batch_size, window, num_features))
            # If your model's output is not directly comparable to targets, adjust as needed
            loss = criterion(outputs, targets)  # Assuming you want the last time step's output
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

num_features = input_dim + external_dim
```
Now, we start the training process.

```python
train_with_early_stopping(model,
                          data_loaders={'train': train_loader, 'val': test_loader},
                          num_epochs=50,
                          window=window,
                          num_features=num_features,
                          batch_size=batch_size,
                          optimizer=None,
                          scheduler=None)
```
Sequential prediction is a method used to predict future values in a time series. 
It involves using the last n data points as the starting point and then using the trained model to forecast future 
values. It will start at the end of the training sequence then forecast num_predictions steps ahead. The forecasted 
values are then plotted against the actual values to visualize the performance of the model.

```python
def sequential_predict(model, X_test, num_predictions, device):
    # Convert to NumPy and remove singleton dimensions
    sequence_to_plot = X_test.squeeze().cpu().numpy()
    # Use the last 30 data points as the starting point
    # sequence_to_plot is 1692 x 30
    # sequence_to_plot[-1] is last row
    historical_data = sequence_to_plot[-1]
    forecasted_values = []

    # Use the trained model to forecast future values
    with torch.no_grad():
        for _ in range(num_predictions * 2):
            # Prepare the historical_data tensor
            # get historical_data to be a tensor then make it 1 x 30 x 1 to get ready for model input
            historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
            # Use the model to predict the next value
            a = model(historical_data_tensor).cpu().numpy()
            predicted_value = a[0, 0]
            # Append the predicted value to the forecasted_values list
            forecasted_values.append(predicted_value[0])

            # Update the historical_data sequence by removing the oldest value and adding the predicted value. Remember the usage here
            historical_data = np.roll(historical_data, shift=-1)
            historical_data[-1] = predicted_value

    return forecasted_values, sequence_to_plot

num_predictions = 30

# get last window of the data
initial_window = train_data[-window:]
predicted_values, sequence_to_plot = sequential_predict(model, X_test, num_predictions, device)

plt.rcParams['figure.figsize'] = [14, 4]
time_val_data = np.arange(0, len(val_data[-100:-30]))
# Test data
plt.plot(time_val_data, val_data[-100:-30], label="test_data", color="b")
# reverse the scaling transformation
original_cases = scaler.inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0)).flatten()
time_original_cases = np.arange(len(val_data[-100:-30]) + 1, len(val_data[-100:-30]) + 1 + len(original_cases))

# the historical data used as input for forecasting
plt.plot(time_original_cases, original_cases, label='actual values', color='green')

time_forecasted_cases = np.arange(len(val_data[-100:-30]) + 1, len(val_data[-100:-30]) + 1 + len(predicted_values))
forecasted_cases = scaler.inverse_transform(np.expand_dims(predicted_values, axis=0)).flatten()
    # plotting the forecasted values
plt.plot(time_forecasted_cases, forecasted_cases, label='forecasted values', color='red')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.title('Time Series Forecasting')
plt.grid(True)
plt.show()
```
![img.png](img.png)