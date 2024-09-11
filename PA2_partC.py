"""
514 Programming Assignment Part C
Group: David, Rashaan, Ravi
First Binary Classification: []
Second Binary Classification:
Third Binary Classification:

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# # One-hot encode the suits of the cards
# def one_hot_encode_suits(data):
#     for i in range(1, 6):  # For each card in the hand
#         data = pd.concat([data, pd.get_dummies(data[f'S{i}'], prefix=f'S{i}')], axis=1).drop([f'S{i}'], axis=1)
#     return data

# Apply preprocessing steps
def preprocess_data(data):
    data = adjust_ranks(data)
    data = adjust_classes(data)
    # data = one_hot_encode_suits(data)
    return data

training_data_preprocessed = preprocess_data(training_data)
testing_data_preprocessed = preprocess_data(testing_data)


#Split Data for Binary classification
    #Set aside 10% of the data for validating the model
from sklearn.model_selection import train_test_split

#Split for each binary classificaitonal problem
training_data_preprocessed_01 = training_data_preprocessed[training_data_preprocessed['Class'] != 2]
testing_data_preprocessed_01 = testing_data_preprocessed[testing_data_preprocessed['Class'] != 2]

training_data_preprocessed_02 = training_data_preprocessed[training_data_preprocessed['Class'] != 1]
testing_data_preprocessed_02 = testing_data_preprocessed[testing_data_preprocessed['Class'] != 1]

training_data_preprocessed_12 = training_data_preprocessed[training_data_preprocessed['Class'] != 0]
testing_data_preprocessed_12 = testing_data_preprocessed[testing_data_preprocessed['Class'] != 0]

#Split each binary classification set into validation and training sets
X_train_01, X_val_01, y_train_01, y_val_01 = train_test_split(training_data_preprocessed_01.drop('Class', axis=1), training_data_preprocessed_01['Class'], test_size=0.1, random_state=42)
X_train_02, X_val_02, y_train_02, y_val_02 = train_test_split(training_data_preprocessed_02.drop('Class', axis=1), training_data_preprocessed_02['Class'], test_size=0.1, random_state=42)
X_train_12, X_val_12, y_train_12, y_val_12 = train_test_split(training_data_preprocessed_12.drop('Class', axis=1), training_data_preprocessed_12['Class'], test_size=0.1, random_state=42)

# Combine the features and target variable for the training set
#First Binary Classification
train_set_01 = pd.concat([X_train_01, y_train_01], axis=1)
val_set_01 = pd.concat([X_val_01, y_val_01], axis=1)


#Second Binary Classification
train_set_02 = pd.concat([X_train_02, y_train_02], axis=1)
val_set_02 = pd.concat([X_val_02, y_val_02], axis=1)

#Third Binary Classification
train_set_12 = pd.concat([X_train_12, y_train_12], axis=1)
val_set_12 = pd.concat([X_val_12, y_val_12], axis=1)

"""Commented out since the data is already downloaded and submitted on cavnas"""
# #Save the training and validation sets to CSV files
# train_set_01.to_csv('514_PA2_partC_training_set_01_first_binary.csv', index=False)
# val_set_01.to_csv('514_PA2_partC_validation_set_01_first_binary.csv', index=False)
# train_set_02.to_csv('514_PA2_partC_training_set_02_second_binary.csv', index=False)
# val_set_02.to_csv('514_PA2_partC_validation_set_02_second_binary.csv', index=False)
# train_set_12.to_csv('514_PA2_partC_training_set_12_third_binary.csv', index=False)
# val_set_12.to_csv('514_PA2_partC_validation_set_12_third_binary.csv', index=False)

#Question 2.2
"""
*From PA2 PDF*
For each model:
    o For each binary classification problem:
        1. Perform 5-fold cross validation on the training dataset
        2. Visualize the cross validation results
        3. Use the best hyperparameter value to train the model on the whole training dataset
        4. Use the trained model to predict on the final validation set
        5. Report the performance and the runtime of steps 3 and 4


Upload visualizations for each model that shows the 5-fold cross-validation 
results from testing different hyperparameter models
"""


"""Decision Tree"""
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree
max_depth_values = [5, 10, 15, 20, None]  # Possible values of max_depth
max_depth_str = [str(md) for md in max_depth_values]  # For plotting
dt_cv_scores = []

for max_depth in max_depth_values:
    dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    scores = cross_val_score(dt_model, X_train_01, y_train_01, cv=kf, scoring='accuracy')
    avg_score = np.mean(scores)
    dt_cv_scores.append(avg_score)
    print(f"Average Accuracy for max_depth={max_depth}: {avg_score:.4f}")

# # Visualize Decision Tree CV results
# plt.figure(figsize=(10, 6))
# plt.plot(max_depth_str, dt_cv_scores, marker='o', linestyle='-', color='blue', label='Decision Tree')
# plt.xlabel('Max Depth')
# plt.ylabel('CV Accuracy')
# plt.title('Decision Tree CV Accuracy by Max Depth')
# plt.grid(True)
# plt.legend()
# plt.show()

"""Random Forest"""
n_estimators_values = [50, 100, 200, 300, 400] #the different hyperparameters to test
rf_cv_scores = []
# Perform 5-fold cross-validation
for n_estimators in n_estimators_values:
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    scores = cross_val_score(rf_model, X_train_01, y_train_01, cv=kf, scoring='accuracy')
    avg_score = np.mean(scores)  # Calculate the average score
    rf_cv_scores.append(avg_score)  # Append the average score to the list
    print(f"Average Accuracy for n_estimators={n_estimators}: {avg_score:.4f}")  

# # Visualize Random Forest CV results
# plt.figure(figsize=(10, 6))
# plt.plot(n_estimators_values, rf_cv_scores, marker='s', linestyle='--', color='green', label='Random Forest')
# plt.xlabel('Number of Estimators')
# plt.ylabel('CV Accuracy')
# plt.title('Random Forest CV Accuracy by Number of Estimators')
# plt.grid(True)
# plt.legend()
# plt.show()


#Question 2.4
# Assuming best_max_depth was determined previously


# best_max_depth = 5  # Example value


# # Initialize Decision Tree model with the best hyperparameter
# dt_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)

# # Measure training runtime
# training_start_time = time.time()
# dt_model.fit(X_train_01, y_train_01)  # Train model on the entire training data
# training_end_time = time.time()

# # Measure prediction runtime
# prediction_start_time = time.time()
# predictions = dt_model.predict(X_val_01)  # Predict on the validation set
# prediction_end_time = time.time()

# #Measure testing runtime
# testing_start_time = time.time()
# test_predictions = dt_model.predict(testing_data_preprocessed_01.drop('Class', axis=1))
# testing_end_time = time.time()

# # Calculate performance
# accuracy = accuracy_score(y_val_01, predictions)
# testing_accuracy = accuracy_score(testing_data_preprocessed_01['Class'], test_predictions)

# # # Report performance and runtime
# # print(f"Decision Tree with max_depth={best_max_depth}:")
# # print(f"Accuracy on validation set: {accuracy:.4f}")
# # print(f"Accuracy on testing set: {testing_accuracy:.4f}")
# # print(f"Training runtime: {training_end_time - training_start_time:.4f} seconds")
# # print(f"Testing runtime: {testing_end_time - testing_start_time:.4f} seconds")
# # print(f"Prediction runtime: {prediction_end_time - prediction_start_time:.4f} seconds")
# # print(f"Testing Prediction runtime: {testing_end_time - testing_start_time:.4f} seconds")

# #Random Forest
# best_n_estimators = 200  # Example value

# rf_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)

# #Measure training runtime
# training_start_time = time.time()
# rf_model.fit(X_train_01, y_train_01)
# training_end_time = time.time()

# #Measure prediction runtime
# prediction_start_time = time.time()
# predictions = rf_model.predict(X_val_01)
# prediction_end_time = time.time()

# #Measure testing runtime
# testing_start_time = time.time()
# test_predictions = rf_model.predict(testing_data_preprocessed_01.drop('Class', axis=1))
# testing_end_time = time.time()

# # Calculate performance
# accuracy = accuracy_score(y_val_01, predictions)
# testing_accuracy = accuracy_score(testing_data_preprocessed_01['Class'], test_predictions)

# # # Report performance and runtime
# # print(f"Random Forest with n_estimators={best_n_estimators}:")
# # print(f"Accuracy on validation set: {accuracy:.4f}")
# # print(f"Accuracy on testing set: {testing_accuracy:.4f}")
# # print(f"Training runtime: {training_end_time - training_start_time:.4f} seconds")
# # print(f"Testing runtime: {testing_end_time - testing_start_time:.4f} seconds")
# # print(f"Prediction runtime: {prediction_end_time - prediction_start_time:.4f} seconds")
# # print(f"Testing Prediction runtime: {testing_end_time - testing_start_time:.4f} seconds")



# #Question 3.1

# # """Decision Tree"""
# # kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # # Decision Tree
# # max_depth_values = [5, 10, 15, 20, None]  # Possible values of max_depth
# # max_depth_str = [str(md) for md in max_depth_values]  # For plotting
# # dt_cv_scores = []

# # for max_depth in max_depth_values:
# #     dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
# #     scores = cross_val_score(dt_model, X_train_12, y_train_12, cv=kf, scoring='accuracy')
# #     avg_score = np.mean(scores)
# #     dt_cv_scores.append(avg_score)
# #     print(f"Average Accuracy for max_depth={max_depth}: {avg_score:.4f}")

# # # Visualize Decision Tree CV results
# # plt.figure(figsize=(10, 6))
# # plt.plot(max_depth_str, dt_cv_scores, marker='o', linestyle='-', color='blue', label='Decision Tree')
# # plt.xlabel('Max Depth')
# # plt.ylabel('CV Accuracy')
# # plt.title('Decision Tree CV Accuracy by Max Depth')
# # plt.grid(True)
# # plt.legend()
# # plt.show()

# # """Random Forest"""
# # n_estimators_values = [50, 100, 200, 300, 400] #the different hyperparameters to test
# # rf_cv_scores = []
# # # Perform 5-fold cross-validation
# # for n_estimators in n_estimators_values:
# #     rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
# #     scores = cross_val_score(rf_model, X_train_12, y_train_12, cv=kf, scoring='accuracy')
# #     avg_score = np.mean(scores)  # Calculate the average score
# #     rf_cv_scores.append(avg_score)  # Append the average score to the list
# #     print(f"Average Accuracy for n_estimators={n_estimators}: {avg_score:.4f}")  

# # # Visualize Random Forest CV results
# # plt.figure(figsize=(10, 6))
# # plt.plot(n_estimators_values, rf_cv_scores, marker='s', linestyle='--', color='green', label='Random Forest')
# # plt.xlabel('Number of Estimators')
# # plt.ylabel('CV Accuracy')
# # plt.title('Random Forest CV Accuracy by Number of Estimators')
# # plt.grid(True)
# # plt.legend()
# # plt.show()



# best_max_depth = 5  # Example value

# # Initialize Decision Tree model with the best hyperparameter
# dt_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)

# # Measure training runtime
# training_start_time = time.time()
# dt_model.fit(X_train_12, y_train_12)  # Train model on the entire training data
# training_end_time = time.time()

# # Measure prediction runtime
# prediction_start_time = time.time()
# predictions = dt_model.predict(X_val_12)  # Predict on the validation set
# prediction_end_time = time.time()

# #Measure testing runtime
# testing_start_time = time.time()
# test_predictions = dt_model.predict(testing_data_preprocessed_12.drop('Class', axis=1))
# testing_end_time = time.time()

# # Calculate performance
# accuracy = accuracy_score(y_val_12, predictions)
# testing_accuracy = accuracy_score(testing_data_preprocessed_12['Class'], test_predictions)

# # Report performance and runtime
# print(f"Decision Tree with max_depth={best_max_depth}:")
# print(f"Accuracy on validation set: {accuracy:.4f}")
# print(f"Training runtime: {training_end_time - training_start_time:.4f} seconds")
# print(f"Prediction runtime: {prediction_end_time - prediction_start_time:.4f} seconds")
# print(f"Accuracy on testing set: {testing_accuracy:.4f}")
# print(f"Testing runtime: {testing_end_time - testing_start_time:.4f} seconds")
# print(f"Testing Prediction runtime: {testing_end_time - testing_start_time:.4f} seconds")

# #Random Forest
# best_n_estimators = 100  # Example value

# rf_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)

# #Measure training runtime
# training_start_time = time.time()
# rf_model.fit(X_train_12, y_train_12)
# training_end_time = time.time()

# #Measure prediction runtime
# prediction_start_time = time.time()
# predictions = rf_model.predict(X_val_12)
# prediction_end_time = time.time()

# #Measure testing runtime
# testing_start_time = time.time()
# test_predictions = rf_model.predict(testing_data_preprocessed_12.drop('Class', axis=1))
# testing_end_time = time.time()

# # Calculate performance
# accuracy = accuracy_score(y_val_12, predictions)
# testing_accuracy = accuracy_score(testing_data_preprocessed_12['Class'], test_predictions)

# # Report performance and runtime
# print(f"Random Forest with n_estimators={best_n_estimators}:")
# print(f"Accuracy on validation set: {accuracy:.4f}")
# print(f"Training runtime: {training_end_time - training_start_time:.4f} seconds")
# print(f"Prediction runtime: {prediction_end_time - prediction_start_time:.4f} seconds")
# print(f"Accuracy on testing set: {testing_accuracy:.4f}")
# print(f"Testing runtime: {testing_end_time - testing_start_time:.4f} seconds")
# print(f"Testing Prediction runtime: {testing_end_time - testing_start_time:.4f} seconds")

