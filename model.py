import tensorflow as tf
from tf.keras.models import Model
from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense



class CNN8LSTM(Model):
    def __init__(self, filters=32, units=50, activation='relu', fps = 3, **kwargs):
        super(CNN8LSTM.self).__init__(**kwargs)

        self.conv1 = TimeDistributed(Conv2D(filters, (3, 3), activation=activation), input_shape=(fps*3, 256, 256, 1))
        self.maxpool = TimeDistributed(MaxPooling2D((2, 2)))
        self.flatten = TimeDistributed(Flatten())

        self.lstm = LSTM(units)
        self.dense1 = Dense(100, activation=activation)
        self.output = Dense(1, activation='sigmoid') # 이진 분류 수행

    def call(self, inputs):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.flatten(x)

        x = self.lstm(x)
        x = self.dense1(x)
        
        return self.output(x)


model = CNN8LSTM()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
