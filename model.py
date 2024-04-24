import tensorflow as tf
from keras.models import Model
from keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense

class CNNLSTM(Model):
    def __init__(self, filters=32, units=50, activation='relu', fps=3, input_shape=(None, 3, 256, 256, 1), **kwargs):
        super().__init__(**kwargs)
        self.fps = fps
        
        # Define the CNN part
        self.conv1 = TimeDistributed(Conv2D(filters, (3, 3), activation=activation),input_shape=input_shape)
        self.maxpool = TimeDistributed(MaxPooling2D((2, 2)))
        self.flatten = TimeDistributed(Flatten())
        
        # Define the LSTM part
        self.lstm = LSTM(units)
        self.dense1 = Dense(100, activation=activation)
        self.dense2 = Dense(1, activation='sigmoid')  # Binary classification output
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x

input_shape = (None, 3, 256, 256, 1)
model = CNNLSTM(input_shape=input_shape)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.build(input_shape=input_shape) 

fake_input = tf.random.normal([5, 3, 256, 256, 1])
_ = model(fake_input)

# 모델 요약 출력
model.summary()