#%% md
# # KKUI CIT Course - Neural networks - Week_02 - Linear regression (basic)
#%%
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#%% md
# ## Prepare data
# This line separates the data into two variables: X contains the features (attributes) of the dataset, while y contains the corresponding labels (target variable). X is a 2-dimensional array where each row represents a sample and each column represents a feature. y is a 1-dimensional array containing the labels for each sample.
# 
# Overall, this code loads the Breast Cancer dataset and separates it into features (X) and labels (y), which are then ready for further processing such as training a neural network model.
#%%
# Loading the Breast Cancer dataset
my_data = datasets.load_breast_cancer()
# Separating features (X) and labels (y) from the dataset
X, y = my_data.data, my_data.target
#%%
# Show out input data
X
#%%
y
#%%
my_data.keys()
#%%
my_data["feature_names"]
#%% md
# ## Data visualization
# 
# ChatGPT
# Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) are both popular dimensionality reduction techniques, but they have different underlying principles and are suitable for different tasks.
# 
# PCA (Principal Component Analysis):
# PCA is a linear dimensionality reduction technique that seeks to find the directions (principal components) in which the data has the maximum variance.
# It transforms the original features into a new set of orthogonal features (principal components) that are linear combinations of the original features.
# The principal components are ordered by the amount of variance they explain in the data, so the first few components capture the most variance.
# PCA is useful for reducing the dimensionality of data while preserving as much variance as possible. It is widely used for data compression, visualization, and noise reduction.
# t-SNE (t-Distributed Stochastic Neighbor Embedding):
# t-SNE is a non-linear dimensionality reduction technique that focuses on preserving the local structure of the data.
# It transforms the high-dimensional data into a lower-dimensional space (typically 2D or 3D) by modeling the similarity of data points as probabilities in both the original and reduced dimensions.
# t-SNE aims to represent similar data points as nearby points in the low-dimensional space while dissimilar points are represented as distant points.
# It is particularly effective for visualizing high-dimensional data clusters and uncovering the local relationships between data points.
# t-SNE is commonly used for exploratory data analysis, visualization, and clustering.
#%%
import plotly.graph_objs as go
import plotly.io as pio

# Apply PCA to reduce data to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get target names
target_names = my_data["target_names"]

# Create hover text for labels
hover_text = [f'Target: {target_names[label]}<br>Attributes (only first 5): {", ".join(map(str, attrs[0:5]))}'
              for label, attrs in zip(y, X)]

# Create trace for data points
trace = go.Scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    mode='markers',
    hovertext=hover_text,
    marker=dict(
        size=7,
        color=y,
        colorscale='Viridis',
        line=dict(
            color='rgb(0, 0, 0)',
            width=0.5
        ),
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='Data Distribution in 2D (PCA)',
    xaxis=dict(title='Principal Component 1'),
    yaxis=dict(title='Principal Component 2'),
    width=800,  # Set the width of the plot
    height=600,  # Set the height of the plot
)

# Create figure
fig = go.Figure(data=[trace], layout=layout)

# Show interactive plot
pio.show(fig)

#%%
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.manifold import TSNE

# Apply t-SNE to reduce data to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Get target names
target_names = my_data["target_names"]

# Create hover text for labels
hover_text = [f'Target: {target_names[label]}<br>Attributes (only first 5): {", ".join(map(str, attrs[0:5]))}'
              for label, attrs in zip(y, X)]

# Create trace for data points
trace = go.Scatter(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    mode='markers',
    hovertext=hover_text,
    marker=dict(
        size=7,
        color=y,
        colorscale='Viridis',
        line=dict(
            color='rgb(0, 0, 0)',
            width=0.5
        ),
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='Data Distribution in 2D (t-SNE)',
    xaxis=dict(title='Component 1'),
    yaxis=dict(title='Component 2'),
    width=800,  # Set the width of the plot
    height=600,  # Set the height of the plot
)

# Create figure
fig = go.Figure(data=[trace], layout=layout)

# Show interactive plot
pio.show(fig)

#%%
import plotly.graph_objs as go
import plotly.io as pio

# Apply PCA to reduce data to 3 dimensions
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Get target names
target_names = my_data["target_names"]

# Create hover text for labels
hover_text = [f'Target: {target_names[label]}<br>Attributes (only first 5): {", ".join(map(str, attrs[0:5]))}'
              for label, attrs in zip(y, X)]

# Create trace for data points
trace = go.Scatter3d(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    z=X_pca[:, 2],
    mode='markers',
    hovertext=hover_text,
    marker=dict(
        size=5,
        color=y,
        colorscale='Viridis',
        line=dict(
            color='rgb(0, 0, 0)',
            width=0.5
        ),
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='Data Distribution in 3D (PCA)',
    scene=dict(
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2'),
        zaxis=dict(title='Principal Component 3')
    ),
    width=1000,  # Set the width of the plot
    height=1000  # Set the height of the plot
)

# Create figure
fig = go.Figure(data=[trace], layout=layout)

# Show interactive plot
pio.show(fig)

#%%
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.manifold import TSNE

# Apply t-SNE to reduce data to 3 dimensions
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X)

# Get target names
target_names = my_data["target_names"]

# Create hover text for labels
hover_text = [f'Target: {target_names[label]}<br>Attributes (only first 5): {", ".join(map(str, attrs[0:5]))}'
              for label, attrs in zip(y, X)]

# Create trace for data points
trace = go.Scatter3d(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    z=X_tsne[:, 2],
    mode='markers',
    hovertext=hover_text,
    marker=dict(
        size=5,
        color=y,
        colorscale='Viridis',
        line=dict(
            color='rgb(0, 0, 0)',
            width=0.5
        ),
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='Data Distribution in 3D (t-SNE)',
    scene=dict(
        xaxis=dict(title='Component 1'),
        yaxis=dict(title='Component 2'),
        zaxis=dict(title='Component 3')
    ),
    width=1000,  # Set the width of the plot
    height=1000  # Set the height of the plot
)

# Create figure
fig = go.Figure(data=[trace], layout=layout)

# Show interactive plot
pio.show(fig)

#%% md
# ### Data splitting
# This line calculates the number of samples and features in the dataset. X.shape returns a tuple where the first element represents the number of samples (rows) and the second element represents the number of features (columns) in the dataset. By unpacking this tuple into n_samples and n_features, you can easily access these values for further processing.
#%%
# Getting the number of samples and features in the dataset
n_samples, n_features = X.shape
#%% md
# This line splits the dataset into training and testing sets using the train_test_split() function from the sklearn.model_selection module. It takes four main arguments:
# - X: The features of the dataset.
# - y: The labels of the dataset.
# - test_size: The proportion of the dataset to include in the test split. Here, it's set to 0.2, meaning 20% of the data will be used for testing.
# - random_state: Controls the shuffling applied to the data before splitting. It ensures reproducibility of the split. Setting a specific random_state (in this case, 1234) ensures that the same random split is obtained each time you run the code.
# 
# After this line executes, you'll have four sets of data:
# 
# X_train: The features of the training set.
# X_test: The features of the testing set.
# y_train: The labels of the training set.
# y_test: The labels of the testing set.
#%%
# Splitting the dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1234)

#%% md
# ### Data normalization
# Normalization is the process of scaling individual samples to have a mean of 0 and a standard deviation of 1. This is important for many machine learning algorithms, including neural networks, because it ensures that features are on a similar scale.
# 
# StandardScaler is a class from the sklearn.preprocessing module that performs this normalization.
# 
# sc.fit_transform(X_train): This line fits the StandardScaler instance sc to the training data X_train and transforms it. This means that it computes the mean and standard deviation of each feature in X_train and then transforms X_train based on these statistics.
# sc.transform(X_test): This line applies the transformation computed from the training data to the testing data X_test. It uses the mean and standard deviation calculated during the fitting step on the training data.
#%%
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_temp = sc.transform(X_temp)
#%%
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure and axes
fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(15, 8))

# Plot histograms for data before normalization
for i in range(0,6):
    sns.histplot(X[:, i], ax=axs[0, i], kde=True, color='blue', alpha=0.5)
    axs[0, i].set_title(f' {my_data["feature_names"][i+1]} (Bef. Norm.)')
    axs[0, i].set_xlabel('Value')
    axs[0, i].set_ylabel('Frequency')

# Plot histograms for data after normalization
for i in range(0,6):
    sns.histplot(X_train[:, i], ax=axs[1, i], kde=True, color='orange', alpha=0.5)
    axs[1, i].set_title(f' {my_data["feature_names"][i+1]} (After. Norm.)')
    axs[1, i].set_xlabel('Value')
    axs[1, i].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

#%% md
# ### Data transformation -> numpy to torch
#%%
# Convert numpy arrays to PyTorch tensors for training, validation, and testing data
X_train = torch.from_numpy(X_train.astype(np.float32))
X_val = torch.from_numpy(X_val.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_val = torch.from_numpy(y_val.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape target tensors to have shape (batch_size, 1)
y_train = y_train.view(y_train.shape[0], 1)
y_val = y_val.view(y_val.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
#%% md
# ## Model
# The `MyModelLogistic` class defines the logistic regression model. It inherits from `nn.Module`, the base class for all neural network modules in PyTorch.
# 
# - `__init__()` method initializes the model. It takes the number of input features as an argument (`n_input_features`).
#   - Inside `__init__`, a linear layer (`self.linear`) is defined using `nn.Linear`. It maps the input features to a single output. This layer represents the equation $f = wx + b$, where `w` are the weights, `x` is the input, and `b` is the bias.
# - `forward()` method defines the forward pass of the model. It takes input tensor `x` and applies the linear transformation followed by the sigmoid activation function. The result is the predicted output (`y_pred`).
# 
# ### Model Initialization:
# 
# The number of samples and features in the dataset (`n_samples`, `n_features`) are obtained from the shape of the input data (`X.shape`).
# 
# An instance of `MyModelLogistic` is created (`model`) with `n_features` passed as the number of input features.
# 
# ### Parameter Access:
# 
# The code iterates over the parameters of the model using `model.parameters()`.
# 
# Inside the loop, each parameter (weights and bias) is printed. These parameters are initialized randomly and will be updated during training to minimize the loss function.
# 
#%%
# Linear model f = wx + b , sigmoid at the end
class MyModelLogistic(nn.Module):
    def __init__(self, n_input_features):
        super(MyModelLogistic, self).__init__()
        # Define a linear layer with input size n_input_features and output size 1
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        # Perform the linear transformation followed by the sigmoid activation function
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# Get the number of samples and features in the dataset
n_samples, n_features = X.shape

# Create an instance of the MyModelLogistic class with the number of input features as the argument
model = MyModelLogistic(n_features)

# Accessing parameters
for param in model.parameters():
    # Print each parameter of the model (weights and bias)
    print(param)

#%% md
# ## Loss and optimizer
#%%
num_epochs = 1000
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#%% md
# ## Train loop
#%%
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # Forward pass and loss for training set
    model.train()  # Set the model to training mode
    train_outputs = model(X_train)
    train_loss = criterion(train_outputs, y_train)

    # Backward pass and update
    train_loss.backward()
    optimizer.step()

    # Zero gradients before new step
    optimizer.zero_grad()

    # Calculate training accuracy
    train_predictions = torch.round(train_outputs)
    train_correct = (train_predictions == y_train).sum().item()
    train_acc = train_correct / len(y_train)

    # Forward pass for validation set
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to compute gradients for validation
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

        # Calculate validation accuracy
        val_predictions = torch.round(val_outputs)
        val_correct = (val_predictions == y_val).sum().item()
        val_acc = val_correct / len(y_val)

    # Store losses and accuracies
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Logging
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}')

#%%
import matplotlib.pyplot as plt

# Plotting training and validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.title(f'Training and Validation Accuracy\nOptimizer: {optimizer.__class__.__name__}, Learning Rate: {learning_rate}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title(f'Training and Validation Loss\nOptimizer: {optimizer.__class__.__name__}, Learning Rate: {learning_rate}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#%% md
# ## Test loop
#%%
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
#%%
