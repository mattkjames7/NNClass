from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class NNClass(object):
	
	def __init__(self,s,AF='sigmoid',Output='softmax',Loss='categorical_crossentropy'):
	
		self.L = np.size(s)
		self.s = s

		#define the activation functions
		if AF == 'LeakyReLU':
			HidAF = layers.LeakyReLU()
		else:
			HidAF = AF
		
		#now try and create a model
		inputs = keras.Input(shape=(s[0],))
		prev = inputs
		for i in range(1,self.L-1):
			x = layers.Dense(s[i],activation=HidAF)(prev)
			prev = x
		outputs = layers.Dense(s[-1],activation=Output)(prev)
		self.model = keras.Model(inputs=inputs,outputs=outputs)
		self.model.compile(optimizer='Adam',loss=Loss,metrics=[Loss,'accuracy'])	
		self.val = None
		self.Jt = np.array([],dtype='float32')
		self.Jc = np.array([],dtype='float32')
		self.At = np.array([],dtype='float32')
		self.Ac = np.array([],dtype='float32')

	def _ClassLabelToOnehot(self,y0):
		'''
		Converts an array of class labels (starting at 1, ending at s[-1])
		to a one-hot 2D array.
		
		'''
		m = np.size(y0)
		n = self.s[-1]
		yoh = np.zeros((m,n),dtype='int32')
		yoh[(np.arange(m),y0-1)] = 1
		return yoh
		
	def AddData(self,X,y):
		'''
		Input shape (m,n)
		'''
		if np.size(X.shape) == 1:
			self.X = np.array([X]).T
		else:
			self.X = np.array(X)
		
		if np.size(y.shape) == 1:
			self.y = self._ClassLabelToOnehot(y)
		else:
			self.y = np.array(y)
			

	def AddValidationData(self,X,y):
		'''
		Input shape (m,n)
		'''
		if np.size(X.shape) == 1:
			self.Xcv = np.array([X]).T
		else:
			self.Xcv = np.array(X)
		
		if np.size(y.shape) == 1:
			self.ycv = self._ClassLabelToOnehot(y)
		else:
			self.ycv = np.array(y)
		self.val = (self.Xcv,self.ycv)
			
		
	def Train(self,nEpoch,BatchSize=1000,verbose=1):
			
		self.hist = self.model.fit(self.X,self.y,epochs=nEpoch,batch_size=BatchSize,validation_data=self.val,verbose=verbose)
		print(self.hist.history.keys())
		self.Jt = np.append(self.Jt,self.hist.history['loss'])
		self.Jc = np.append(self.Jc,self.hist.history['val_loss'])
		self.At = np.append(self.At,self.hist.history['acc'])
		self.Ac = np.append(self.Ac,self.hist.history['val_acc'])
		return self.hist
		
	def Predict(self,X):
		if np.size(X.shape) == 1:
			x = np.array([X]).T
		else:
			x = X
		y = self.model.predict(x)

		return y
	
	
	def GetWeights(self):
		w = []
		b = []
		tmp = self.model.get_weights()
		for i in range(0,self.L-1):
			w.append(tmp[i*2])
			b.append(tmp[i*2+1])
		return w,b
		
	def SetWeights(self,w,b):
		ipt = []
		for i in range(0,self.L-1):
			ipt.append(w[i])
			ipt.append(b[i])
		self.model.set_weights(ipt)
