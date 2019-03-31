#! usr/bin/env python
# *-- coding : utf-8 --*


from __future__ import division
import codecs
import sys
import numpy as np
import pandas as pd
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping