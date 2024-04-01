import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

  
"""
514 Programming Assignment 2
Group: Rashaan, Ravi, David

"""
"""
Data additional information
Each record is an example of a hand consisting of five playing cards drawn from a standard deck of 52. 
Each card is described using two attributes (suit and rank), for a total of 10 predictive attributes. 
There is one Class attribute that describes the "Poker Hand". 
The order of cards is important, which is why there are 480 possible Royal Flush hands as compared to 4
"""

"""Load dataset"""
# Column names for the dataset
# Define the column names for your dataset
columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class']

# Load your dataset
training_data = pd.read_csv('poker-hand-training-true.csv', header=None, names=columns)
testing_data = pd.read_csv('poker-hand-testing.csv', header=None, names=columns)

# Adjust the ranks of cards, specifically changing Ace from 1 to 14
def adjust_ranks(data):
    for i in range(1, 6):  # For each card in the hand
        data[f'C{i}'] = data[f'C{i}'].apply(lambda x: 14 if x == 1 else x)
    return data

# One-hot encode the suits of the cards
def one_hot_encode_suits(data):
    for i in range(1, 6):  # For each card in the hand
        data = pd.concat([data, pd.get_dummies(data[f'S{i}'], prefix=f'S{i}')], axis=1).drop([f'S{i}'], axis=1)
    return data

# Apply preprocessing steps
def preprocess_data(data):
    data = adjust_ranks(data)
    data = one_hot_encode_suits(data)
    return data

training_data_preprocessed = preprocess_data(training_data)
testing_data_preprocessed = preprocess_data(testing_data)

# #Final Preprocessed DATA
# print(training_data_preprocessed.head())
# print(testing_data_preprocessed.head())

"""Decision Tree"""
#using traingin dataset
X_train = training_data_preprocessed.drop('Class', axis=1)
y_train = training_data_preprocessed['Class']

# Set up KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List to store results for different hyperparameters
max_depth_values = [5, 10, 15, 20, None]  # None means no limit on the depth of the tree
avg_scores = []

# Loop over hyperparameter values
for max_depth in max_depth_values:
    # Initialize the model with the current hyperparameter value
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    # Perform 5-fold cross-validation and store the scores
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    
    # Compute the average accuracy and store it
    avg_score = np.mean(scores)
    avg_scores.append(avg_score)
    
    # Print the results
    print(f"Average Accuracy for max_depth={max_depth}: {avg_score:.4f}")

# Find the best hyperparameter value
best_index = np.argmax(avg_scores)
best_max_depth = max_depth_values[best_index]
print(f"Best max_depth value: {best_max_depth}")

#Random Forest
# Define hyperparameters to test
n_estimators_values = [50, 100, 200, 300]
avg_scores_rf = []

# Loop over hyperparameter values
for n_estimators in n_estimators_values:
    # Initialize the model with the current hyperparameter value
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    # Perform 5-fold cross-validation and store the scores
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')
    
    # Compute the average accuracy and store it
    rf_avg_score = np.mean(rf_scores)
    avg_scores_rf.append(rf_avg_score)
    
    print(f"Average Accuracy for n_estimators={n_estimators}: {rf_avg_score:.4f}")

# Find the best hyperparameter value
best_index_rf = np.argmax(avg_scores_rf)
best_n_estimators = n_estimators_values[best_index_rf]
print(f"Best n_estimators value for Random Forest: {best_n_estimators}")


# #PLots
# # Plot for Decision Tree
# plt.figure(figsize=(10, 5))
# plt.plot(max_depth_values, avg_scores, marker='o', linestyle='-', color='b', label='Decision Tree')

# # Plot for Random Forest
# plt.plot(n_estimators_values, avg_scores_rf, marker='s', linestyle='--', color='g', label='Random Forest')

# plt.xlabel('Hyperparameter Value')
# plt.ylabel('Average Accuracy')
# plt.title('5-Fold Cross-Validation Results')
# plt.legend()
# plt.grid(True)

# plt.show()