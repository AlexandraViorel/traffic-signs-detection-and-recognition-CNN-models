from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, RandomFlip, RandomRotation

class TSRNet3New():
    def build_model(k, f1, f2, f3, fc_layer, drop):
        model = Sequential()
        model.add(Conv2D(filters=f1, kernel_size=(k,k), padding='same', activation='leaky_relu', input_shape=(45, 45, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(filters=f1, kernel_size=(k,k), padding='same', activation='leaky_relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(filters=f1, kernel_size=(k,k), padding='same', activation='leaky_relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=f2, kernel_size=(k,k), padding='same', activation='leaky_relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(filters=f2, kernel_size=(k,k), padding='same', activation='leaky_relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=f3, kernel_size=(k,k), padding='same', activation='leaky_relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(fc_layer, activation='leaky_relu'))
        model.add(Dropout(rate=drop))
        model.add(Dense(fc_layer, activation='leaky_relu'))
        model.add(Dropout(rate=drop))
        model.add(Dense(75, activation='softmax'))

        return model