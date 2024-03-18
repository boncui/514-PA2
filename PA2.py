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

"""Create Visualizations of Variables"""