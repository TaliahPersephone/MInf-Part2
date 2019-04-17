import keras

class CombinedModel(keras.Model):
	def __init__(self,batch_size = 623):
		super(CombinedModel,self).__init__(name='combined')
		self.rshape = keras.layers.Reshape((64,64,1))
		self.rshape.build((batch_size,64*64))

		self.cnn1 = keras.layers.Conv2D(20, kernel_size=5,padding='same',activation='relu',kernel_initializer='random_normal')
		self.cnn1.build((batch_size,64,64,1))

		self.cnnbn1 = keras.layers.BatchNormalization()
		self.cnnbn1.build((batch_size,64,64,20))

		self.mp1 = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))
		self.mp1.build((batch_size,64,64,20))
	
		self.cnn2 = keras.layers.Conv2D(20, kernel_size=3,padding='same',activation='relu',kernel_initializer='random_normal')
		self.cnn2.build((batch_size,32,32,20))
		
		self.cnnbn2 = keras.layers.BatchNormalization()
		self.cnnbn2.build((batch_size,32,32,400))

		self.mp2 = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))
		self.mp2.build((batch_size,32,32,400))

		self.flat = keras.layers.Flatten()
		self.flat.build((batch_size,16,16,400))

		self.cnnd = keras.layers.Dense(512,activation='relu',kernel_initializer='random_normal')
		self.cnnd.build((batch_size,102400))

		self.histd = keras.layers.Dense(512,activation='relu',kernel_initializer='random_normal')
		self.histd.build((batch_size,6272))

		self.conc = keras.layers.Concatenate()
		self.conc.build([(batch_size,512),(batch_size,512)])

		self.bn1 = keras.layers.BatchNormalization()
		self.bn1.build((batch_size,1024))

		self.lstm1 = keras.layers.LSTM(256, return_sequences=True,activation='relu',kernel_initializer='random_normal')
		self.lstm1.build((batch_size,1024))
		
		self.bn2 = keras.layers.BatchNormalization()
		self.bn2.build((batch_size,1,256))

		self.lstm2 = keras.layers.LSTM(1, return_sequences=False,activation='sigmoid',kernel_initializer='random_normal')
		self.lstm2.build((batch_size,1,256))

	def call(self,inputs):
		x = inputs[:64*64]
		y = inputs[64*64:]

		x = self.rshape(x)
		x = self.cnn1(x)
		x = self.cnnbn1(x)
		x = self.mp1(x)
		x = self.cnn2(x)
		x = self.cnnbn2(x)
		x = self.mp2(x)
		x = self.flat(x)
		x = self.cnnd(x)

		y = self.histd(y)

		x = self.conc(x,y)

		x = self.bn1(x)
		x = self.lstm1(x)
		x = self.bn2(x)
		
		return self.lstm2(x)
		
		
