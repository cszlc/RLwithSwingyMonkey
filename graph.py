"""
Script to make graphs based on the data saved in the .npy files.
Displays score by epoch of the Q-learning algorithm.
"""
import numpy as np
import matplotlib.pyplot as plt

# Load the numpy array from the .npy file
histogram = np.load('hist.npy')

# Plot the histogram
plt.plot(histogram)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Score by Epoch (Gam = 0.75, LR = 0.8, Init_Eps = 0.1, Dec_Rate = 10%)")
plt.show()
