from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

class ELM(object):
    
    def __init__(self, input_layer = 'relu', hidden_layer = ['relu'], output_layer = 'relu'):
    	self.input_layer = input_layer
    	self.hidden_layer = hidden_layer
    	self.output_layer = output_layer
    	self.model = None

    def buildNetwork(self):
    	pass

    def setWeights(self):
		weights = self.model.layers[-1].get_weights()
		weights[0] = np.random.random_sample((weights[0].shape[0],weights[0].shape[1]))
		self.model.layers[-1].set_weights(weights)

    def fit(self, X_train, y_train):
		self.model.fit(X_train, y_train, verbose=2, batch_size=5, epochs=200)

    def evaluete(self, X_test, y_test):
		return self.model.evaluate(X_test, y_test)

    def getWeights(self):
		return self.model.get_weights()

    def summary(self):
		self.model.summary()

class MultiClassify(ELM):

	def buildNetwork(self, input_dim):
		self.model = Sequential()

		# Input layer
		self.model.add(Dense(units=100, activation=self.input_layer, input_dim=input_dim, trainable=False))
		self.setWeights()
		# Hidden layer
		for act in self.hidden_layer:
			self.model.add(Dense(units=10, activation=act, trainable=True))
		# Output layer
		self.model.add(Dense(units=3, activation=self.output_layer, trainable=True))

		self.model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

class BinaryClassify(ELM):

	def buildNetwork(self, input_dim):
		self.model = Sequential()

		# Input layer
		self.model.add(Dense(units=110, activation=self.input_layer, input_dim=input_dim, trainable=False))
		self.setWeights()
		# Hidden layer
		for act in self.hidden_layer:
			self.model.add(Dense(units=10, activation=act, trainable=True))
		# Output layer
		self.model.add(Dense(units=2, activation=self.output_layer, trainable=True))

		self.model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])


	

	

	


