"""
514 Programming Assignment Part D
Group: David, Rashaan, Ravi
First Binary Classification: []
Second Binary Classification:
Third Binary Classification:

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time



#import data
columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class']
training_data= pd.read_csv('poker-hand-training-true.csv', header=None, names=columns)
testing_data = pd.read_csv('poker-hand-testing.csv', header=None, names=columns)

#Load Prerpocessing Data

# Adjust the ranks of cards, specifically changing Ace from 1 to 14
def adjust_ranks(data):
    for i in range(1, 6):  # For each card in the hand
        data[f'C{i}'] = data[f'C{i}'].apply(lambda x: 14 if x == 1 else x)
    return data

# Make all rows with classes 0-3 to be 0, 4-6 to be 1, and 7-9 to be 2
def adjust_classes(data):
    data['Class'] = data['Class'].apply(lambda x: 0 if x < 4 else 1 if x < 7 else 2)
    return data

# One-hot encode the suits of the cards
def one_hot_encode_suits(data):
    for i in range(1, 6):  # For each card in the hand
        data = pd.concat([data, pd.get_dummies(data[f'S{i}'], prefix=f'S{i}')], axis=1).drop([f'S{i}'], axis=1)
    return data

# Apply preprocessing steps
def preprocess_data(data):
    data = adjust_ranks(data)
    data = adjust_classes(data)
    data = one_hot_encode_suits(data)
    return data

training_data_preprocessed = preprocess_data(training_data)
testing_data_preprocessed = preprocess_data(testing_data)


#Split for each binary classificaitonal problem
training_data_preprocessed_01 = training_data_preprocessed[training_data_preprocessed['Class'] != 2]
testing_data_preprocessed_01 = testing_data_preprocessed[testing_data_preprocessed['Class'] != 2]

training_data_preprocessed_02 = training_data_preprocessed[training_data_preprocessed['Class'] != 1]
testing_data_preprocessed_02 = testing_data_preprocessed[testing_data_preprocessed['Class'] != 1]

training_data_preprocessed_12 = training_data_preprocessed[training_data_preprocessed['Class'] != 0]
testing_data_preprocessed_12 = testing_data_preprocessed[testing_data_preprocessed['Class'] != 0]

#Split each binary classification set into validation and training sets
X_train_01, X_val_01, y_train_01, y_val_01 = train_test_split(
    training_data_preprocessed_01.drop('Class', axis=1), training_data_preprocessed_01['Class'], test_size=0.1, random_state=42)
X_train_02, X_val_02, y_train_02, y_val_02 = train_test_split(
    training_data_preprocessed_02.drop('Class', axis=1), training_data_preprocessed_02['Class'], test_size=0.1, random_state=42)
X_train_12, X_val_12, y_train_12, y_val_12 = train_test_split(
    training_data_preprocessed_12.drop('Class', axis=1), training_data_preprocessed_12['Class'], test_size=0.1, random_state=42)

# Function to apply PCA
def apply_and_transform_pca(X_train, X_val, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca

# Apply PCA to the datasets
X_train_01_pca, X_val_01_pca, X_test_01_pca = apply_and_transform_pca(X_train_01, X_val_01, testing_data_preprocessed_01.drop('Class', axis=1))
X_train_02_pca, X_val_02_pca, X_test_02_pca = apply_and_transform_pca(X_train_02, X_val_02, testing_data_preprocessed_02.drop('Class', axis=1))
X_train_12_pca, X_val_12_pca, X_test_12_pca = apply_and_transform_pca(X_train_12, X_val_12, testing_data_preprocessed_12.drop('Class', axis=1))



"""Visualization of 5-fold Cross Validation"""

"""Decision Tree"""
#Binary 1
dt_model_01 = DecisionTreeClassifier(max_depth=10, random_state=42)  # Using depth of 10 as specified
dt_model_01.fit(X_train_01_pca, y_train_01)
predictions_01 = dt_model_01.predict(X_val_01_pca)
accuracy_01 = accuracy_score(y_val_01, predictions_01)
print(f"Decision Tree Accuracy on PCA-transformed data (Binary Classification 01): {accuracy_01:.4f}")

#Binary 2
dt_model_02 = DecisionTreeClassifier(max_depth=10, random_state=42)  # Using depth of 10 as specified
dt_model_02.fit(X_train_02_pca, y_train_02)
predictions_02 = dt_model_02.predict(X_val_02_pca)
accuracy_02 = accuracy_score(y_val_02, predictions_02)
print(f"Decision Tree Accuracy on PCA-transformed data (Binary Classification 02): {accuracy_02:.4f}")

#Binary 3
dt_model_12 = DecisionTreeClassifier(max_depth=10, random_state=42)  # Using depth of 10 as specified
dt_model_12.fit(X_train_12_pca, y_train_12)
predictions_12 = dt_model_12.predict(X_val_12_pca)
accuracy_12 = accuracy_score(y_val_12, predictions_12)
print(f"Decision Tree Accuracy on PCA-transformed data (Binary Classification 12): {accuracy_12:.4f}")

#DT Cross validation
dt_cv_accuracies_01 = cross_val_score(DecisionTreeClassifier(max_depth=10, random_state=42),
                                      X_train_01_pca, y_train_01, cv=5, scoring='accuracy')
dt_cv_accuracies_02 = cross_val_score(DecisionTreeClassifier(max_depth=10, random_state=42),
                                      X_train_02_pca, y_train_02, cv=5, scoring='accuracy')
dt_cv_accuracies_12 = cross_val_score(DecisionTreeClassifier(max_depth=10, random_state=42),
                                      X_train_12_pca, y_train_12, cv=5, scoring='accuracy')

print(f"Decision Tree Average CV Accuracy (Binary 01): {np.mean(dt_cv_accuracies_01):.4f}")
print(f"Decision Tree Average CV Accuracy (Binary 02): {np.mean(dt_cv_accuracies_02):.4f}")
print(f"Decision Tree Average CV Accuracy (Binary 12): {np.mean(dt_cv_accuracies_12):.4f}")

"""Random Forest"""
# Binary 1
rf_model_01 = RandomForestClassifier(n_estimators=200, random_state=42)  # Using 200 estimators as specified
rf_model_01.fit(X_train_01_pca, y_train_01)
predictions_01 = rf_model_01.predict(X_val_01_pca)
accuracy_01 = accuracy_score(y_val_01, predictions_01)
print(f"Random Forest Accuracy on PCA-transformed data (Binary Classification 01): {accuracy_01:.4f}")

#Binary 2
rf_model_02 = RandomForestClassifier(n_estimators=200, random_state=42)  # Using 200 estimators as specified
rf_model_02.fit(X_train_02_pca, y_train_02)
predictions_02 = rf_model_02.predict(X_val_02_pca)
accuracy_02 = accuracy_score(y_val_02, predictions_02)
print(f"Random Forest Accuracy on PCA-transformed data (Binary Classification 02): {accuracy_02:.4f}")

#Binary 3  
rf_model_12 = RandomForestClassifier(n_estimators=200, random_state=42)  # Using 200 estimators as specified
rf_model_12.fit(X_train_12_pca, y_train_12)
predictions_12 = rf_model_12.predict(X_val_12_pca)
accuracy_12 = accuracy_score(y_val_12, predictions_12)
print(f"Random Forest Accuracy on PCA-transformed data (Binary Classification 12): {accuracy_12:.4f}")

#RF Cross validation
rf_cv_accuracies_01 = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42),
                                      X_train_01_pca, y_train_01, cv=5, scoring='accuracy')
rf_cv_accuracies_02 = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42),
                                      X_train_02_pca, y_train_02, cv=5, scoring='accuracy')
rf_cv_accuracies_12 = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42),
                                      X_train_12_pca, y_train_12, cv=5, scoring='accuracy')
print(f"Random Forest Average CV Accuracy (Binary 01): {np.mean(rf_cv_accuracies_01):.4f}")
print(f"Random Forest Average CV Accuracy (Binary 02): {np.mean(rf_cv_accuracies_02):.4f}")
print(f"Random Forest Average CV Accuracy (Binary 12): {np.mean(rf_cv_accuracies_12):.4f}")


#Visualizations
# Calculate mean accuracies
# Setup for cross-validation: 5 folds
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Decision Tree: Test different max_depth values
# dt_depths = [5, 10, 15, 20, None]
# dt_scores = {'Binary 01': [], 'Binary 02': [], 'Binary 12': []}

# for depth in dt_depths:
#     dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
#     # Iterate through each binary classification dataset
#     for binary, X, y in zip(['Binary 01', 'Binary 02', 'Binary 12'],
#                             [X_train_01_pca, X_train_02_pca, X_train_12_pca],
#                             [y_train_01, y_train_02, y_train_12]):
#         scores = cross_val_score(dt_model, X, y, cv=kf, scoring='accuracy')
#         # print(f"Decision Tree Average CV Accuracy (Binary {binary}): {scores.mean():.4f}")
#         dt_scores[binary].append(scores.mean())

# # Visualize Decision Tree CV results
# plt.figure(figsize=(10, 6))
# plt.plot(max_depth_str, dt_cv_scores, marker='o', linestyle='-', color='blue', label='Decision Tree')
# plt.xlabel('Max Depth')
# plt.ylabel('CV Accuracy')
# plt.title('Decision Tree CV Accuracy by Max Depth')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Random Forest: Test different n_estimators values
# rf_estimators = [50, 100, 200, 300, 400]
# rf_scores = {'Binary 01': [], 'Binary 02': [], 'Binary 12': []}

# for n_estimators in rf_estimators:
#     rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
#     # Iterate through each binary classification dataset
#     for binary, X, y in zip(['Binary 01', 'Binary 02', 'Binary 12'],
#                             [X_train_01_pca, X_train_02_pca, X_train_12_pca],
#                             [y_train_01, y_train_02, y_train_12]):
#         scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
#         # print(f"Random Forest Average CV Accuracy (Binary {binary}): {scores.mean():.4f}")
#         rf_scores[binary].append(scores.mean())


# # Visualize Random Forest CV results
# plt.figure(figsize=(10, 6))
# plt.plot(n_estimators_values, rf_cv_scores, marker='s', linestyle='--', color='green', label='Random Forest')
# plt.xlabel('Number of Estimators')
# plt.ylabel('CV Accuracy')
# plt.title('Random Forest CV Accuracy by Number of Estimators')
# plt.grid(True)
# plt.legend()
# plt.show()

"""MEasure runtime and performance"""
dt_model_12 = DecisionTreeClassifier(max_depth=10, random_state=42)

# Measure training runtime for Decision Tree
training_start_time = time.time()
dt_model_12.fit(X_train_12_pca, y_train_12)
training_end_time = time.time()

# Measure prediction runtime for Decision Tree
prediction_start_time = time.time()
predictions_02 = dt_model_12.predict(X_val_12_pca)
prediction_end_time = time.time()

# Measure test runtime for Decision Tree
testing_start_time = time.time()
predictions_12_test = dt_model_12.predict(X_test_12_pca)
testing_end_time = time.time()

# Calculate performance for Decision Tree
accuracy_12 = accuracy_score(y_val_12, predictions_12)
testing_accuracy_12 = accuracy_score(testing_data_preprocessed_12['Class'], predictions_12_test)

# Report performance and runtime for Decision Tree
print(f"Decision Tree with max_depth=10:")
print(f"Accuracy on validation set: {accuracy_12:.4f}")
print(f"Accuracy on testing set: {testing_accuracy_12:.4f}")
print(f"Training runtime: {training_end_time - training_start_time:.4f} seconds")
print(f"Testing runtime: {testing_end_time - testing_start_time:.4f} seconds")
print(f"Prediction runtime: {prediction_end_time - prediction_start_time:.4f} seconds")

#Random Forest
rf_model_12 = RandomForestClassifier(n_estimators=200, random_state=42)

# Measure training runtime for Random Forest
training_start_time = time.time()
rf_model_12.fit(X_train_12_pca, y_train_12)
training_end_time = time.time()

# Measure prediction runtime for Random Forest
prediction_start_time = time.time()
predictions_rf_12 = rf_model_12.predict(X_val_12_pca)
prediction_end_time = time.time()

# Measure test runtime for Random Forest
testing_start_time = time.time()
predictions_rf_12_test = rf_model_12.predict(X_test_12_pca)
testing_end_time = time.time()

# Calculate performance for Random Forest
accuracy_rf_12 = accuracy_score(y_val_12, predictions_rf_12)
testing_accuracy_rf_12 = accuracy_score(testing_data_preprocessed_12['Class'], predictions_rf_12_test)

# Report performance and runtime for Random Forest
print(f"Random Forest with n_estimators=200:")
print(f"Accuracy on validation set: {accuracy_rf_12:.4f}")
print(f"Accuracy on testing set: {testing_accuracy_rf_12:.4f}")
print(f"Training runtime: {training_end_time - training_start_time:.4f} seconds")
print(f"Testing runtime: {testing_end_time - testing_start_time:.4f} seconds")
print(f"Prediction runtime: {prediction_end_time - prediction_start_time:.4f} seconds")