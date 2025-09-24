Algorithm:
Step 1: Import Libraries

Import necessary libraries for:

Data Handling: pandas, numpy

Preprocessing & Splitting: sklearn.preprocessing, sklearn.model_selection

PyTorch: torch, torch.nn, torch.optim

Utilities: random, os (for reproducibility)

Step 2: Load and Inspect Data

Load the CSV dataset using pandas.read_csv().

Check for missing values: df.isnull().sum()

Examine data types: df.dtypes

Explore unique values in categorical columns: df[col].unique()

Step 3: Data Preparation
Identify Columns

Categorical features: e.g., workclass, education, marital-status

Numerical features: e.g., age, hours-per-week, capital-gain

Target column: income

Encode Categorical Features

Map categories to integers using a dictionary for each column.

Replace categorical values with corresponding integers.

Normalize Numerical Features

Calculate mean and standard deviation for each numerical feature.

Prepare Labels

Convert income column:

<=50K → 0

>50K → 1

Step 4: Split Data

Split the dataset:

Training set: 25,000 rows

Testing set: 5,000 rows

Use train_test_split() with a fixed random seed for reproducibility.

Convert all splits to NumPy arrays for PyTorch compatibility.

Step 5: Convert to PyTorch Tensors

Categorical arrays → torch.LongTensor

Numerical arrays → torch.FloatTensor

Labels → torch.LongTensor

Step 6: Define the Neural Network Model

Create a class TabularModel (inherits from torch.nn.Module).

Components:

Embedding Layers

One embedding per categorical feature:
nn.Embedding(num_categories, embedding_dim)

Batch Normalization for Continuous Features

nn.BatchNorm1d(num_continuous_features)

Fully Connected Layers

Input size = sum of embedding dimensions + number of numerical features

Hidden layer: 50 neurons → nn.Linear(input_dim, 50)

Activation: nn.ReLU()

Dropout: nn.Dropout(p=0.4)

Output layer: nn.Linear(50, 2) (2 classes)

Forward Pass:

Pass categorical features through embeddings.

Concatenate embeddings with normalized continuous features.

Apply batch normalization.

Feed through hidden and output layers to get logits.

Step 7: Initialize Model, Loss, and Optimizer

Set seed: torch.manual_seed()

Instantiate the TabularModel

Define loss: nn.CrossEntropyLoss()

Define optimizer: torch.optim.Adam(model.parameters(), lr=0.001)

Step 8: Train the Model

Loop for 300 epochs:

Set model to training mode: model.train()

Reset gradients: optimizer.zero_grad()

Forward pass: outputs = model(categorical_train, continuous_train)

Compute loss: loss = criterion(outputs, labels_train)

Backpropagation: loss.backward()

Update weights: optimizer.step()

Optionally print the loss every few epochs to track progress.

Step 9: Evaluate on Test Data

Set model to evaluation mode: model.eval()

Disable gradient calculation: with torch.no_grad():

Forward pass on test set: outputs = model(categorical_test, continuous_test)

Compute loss and accuracy:

Loss: criterion(outputs, labels_test)

Predictions: torch.argmax(outputs, dim=1)

Accuracy: (predictions == labels_test).sum() / total_samples

Step 10: Predict New Inputs

Define a function predict_income(user_input):

Accept a dictionary of new feature values.

Encode categorical values using the same mapping as training.

Normalize continuous values using training statistics.

Convert input to tensors.

Forward pass through the trained model.

Output:

1 → Income > $50K

0 → Income ≤ $50K
