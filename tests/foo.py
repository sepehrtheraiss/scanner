import numpy as np
from scipy.signal import argrelextrema
x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
argrelextrema(x, np.greater, axis=0)
