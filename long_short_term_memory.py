from sklearn import utils
from math import sqrt
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import mean_squared_error as mse, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler