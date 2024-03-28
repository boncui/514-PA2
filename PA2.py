import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 

  
"""
514 Programming Assignment 2
Group: Rashaan, Ravi, David

"""

"""Fetch dataset"""
poker_hand = fetch_ucirepo(id=158) 
  
X = poker_hand.data.features 
y = poker_hand.data.targets 
# metadata 
    # print(poker_hand.metadata) 
  
# variable information 
    # print(poker_hand.variables) 

"""Load dataset"""
columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class']
training_data = pd.read_csv('poker-hand-training-true.csv')
testing_data = pd.read_csv('poker-hand-testing.csv')
print(training_data.head())
print(testing_data.head())


# """Create Visualizations of Variables"""
# """ PA2 PartA """
# # columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class']
# # training_data.columns = columns
# # testing_data.columns = columns

# # # Suits Distribution
# # plt.figure(figsize=(14, 7))
# # for i, suit in enumerate(['S1', 'S2', 'S3', 'S4', 'S5'], 1):
# #     plt.subplot(2, 5, i)
# #     plt.hist(training_data[suit], bins=np.arange(1, 6) - 0.5, rwidth=0.8)
# #     plt.title(f'Distribution of {suit}')
# #     plt.xticks(range(1, 5))

# # # Ranks Distribution
# # for i, rank in enumerate(['C1', 'C2', 'C3', 'C4', 'C5'], 6):
# #     plt.subplot(2, 5, i)
# #     plt.hist(training_data[rank], bins=np.arange(1, 15) - 0.5, rwidth=0.8)
# #     plt.title(f'Distribution of {rank}')
# #     plt.xticks(range(1, 14))

# # plt.tight_layout()
# # plt.show()

# # plt.figure(figsize=(10, 6))
# # plt.hist(testing_data['Class'], bins=np.arange(-0.5, 10.5, 1), rwidth=0.8, color='skyblue')
# # plt.title('Distribution of Poker Hand Classifications')
# # plt.xticks(range(0, 10))
# # plt.xlabel('Poker Hand Class')
# # plt.ylabel('Frequency')
# # plt.show()

# """Preprocess the data into three binary classification problems"""
# # Define column names for convenience
# columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class']

# # Preprocessing function
# def preprocess_for_classification(data):
#     fold_map = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
#     hold_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
#     bet_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    
#     data['Fold'] = data['Class'].map(fold_map)
#     data['Hold'] = data['Class'].map(hold_map)
#     data['Bet'] = data['Class'].map(bet_map)
    
#     return data

# # Apply preprocessing
# training_data_preprocessed = preprocess_for_classification(training_data.copy())
# testing_data_preprocessed = preprocess_for_classification(testing_data.copy())

# # Verify the preprocessing
# print(training_data_preprocessed[['Class', 'Fold', 'Hold', 'Bet']].head())
# print(testing_data_preprocessed[['Class', 'Fold', 'Hold', 'Bet']].head())