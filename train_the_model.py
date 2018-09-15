import pandas as pd
from data_generator import DataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
import numpy as np
if __name__ == '__main__': 
    driving = pd.read_csv('C:\CarND\CarND-Behavioral-Cloning-P3\data\driving_log.csv')

    # Parameters
    params = {'dim': (160, 320),
              'batch_size': 64,
              'n_classes': 1,
              'n_channels': 3,
              'shuffle': True}

    # Generators
    cutting_index = int(len(driving.center) * 0.8)
    X_training = np.array(driving.center[:cutting_index:])
    y_training = np.array(driving.steering_angle[:cutting_index:])

    X_validation = np.array(driving.center[cutting_index::])
    y_validation = np.array(driving.steering_angle[cutting_index::])

    training_generator = DataGenerator(X_training, y_training, **params)
    validation_generator = DataGenerator(X_validation, y_validation, **params)

    # Design model
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255 - 0.5))
    model.add(Flatten())
    model.add(Dense(1))
    #[...] # Architecture
    model.compile(loss='mse', optimizer='adam')

    # Train model on dataset
  
    model.fit_generator(generator = training_generator,
                        validation_data = validation_generator,
                        use_multiprocessing = True,
                        workers = 8,
                        epochs = 5)
    model.save('model.h5')