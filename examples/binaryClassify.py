from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# import ELM Multiclass classify
import sys
sys.path.append('../src/')
from elm import BinaryClassify

# Load data frame
df = load_breast_cancer()
X = df.data.astype(np.float32) 
y = df.target.reshape(-1, 1).astype(np.int32)

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Split data frame
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create classify
elm = BinaryClassify(hidden_layer=['relu', 'linear', 'sigmoid'], output_layer='sigmoid') 

input_dim = X.shape[1] # number of features
elm.buildNetwork(input_dim)

elm.fit(X_train, y_train)

print elm.evaluete(X_test, y_test)[1]
