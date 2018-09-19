import pandas as pd
from data_generator import DataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, ELU
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
    steering_angle_correction = 0.2

    X_training_center = np.array(driving.center[:cutting_index:])
    y_training_center = np.array(driving.steering_angle[:cutting_index:])

    X_training_left = np.array(driving.left[:cutting_index:])
    y_training_left = np.array(driving.steering_angle[:cutting_index:]) + steering_angle_correction

    X_training_right = np.array(driving.right[:cutting_index:])
    y_training_right = np.array(driving.steering_angle[:cutting_index:]) - steering_angle_correction

    X_training = np.concatenate((X_training_center, X_training_left, X_training_right))
    y_training = np.concatenate((y_training_center, y_training_left, y_training_right))

    X_validation_center = np.array(driving.center[cutting_index::])
    y_validation_center = np.array(driving.steering_angle[cutting_index::])
    
    X_validation_left = np.array(driving.left[cutting_index::])
    y_validation_left = np.array(driving.steering_angle[cutting_index::]) + steering_angle_correction

    X_validation_right = np.array(driving.right[cutting_index::])
    y_validation_right = np.array(driving.steering_angle[cutting_index::]) - steering_angle_correction
    
    X_validation = np.concatenate((X_validation_center, X_validation_left, X_validation_right))
    y_validation = np.concatenate((y_validation_center, y_validation_left, y_validation_right))

    training_generator = DataGenerator(X_training, y_training, **params)
    validation_generator = DataGenerator(X_validation, y_validation, **params)

    # Design model
    model = Sequential()
    model.add(Cropping2D(cropping=((10,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255 - 0.5))
    #model.add(Convolution2D(1, 1, strides=(1,1)))
    #model.add(Convolution2D(24, 5, strides=(2,2)))
    #model.add(Dropout(0.2))
    #model.add(ELU())
    #model.add(Convolution2D(36, 5, strides=(2,2)))
    #model.add(Dropout(0.2))
    #model.add(ELU())
    #model.add(Convolution2D(48, 5, strides=(2,2)))
    #model.add(Dropout(0.2))
    #model.add(ELU())
    #model.add(Convolution2D(64, 3))
    #model.add(ELU())
    #model.add(Convolution2D(64, 3))
    #model.add(ELU())
    model.add(Flatten())
    #model.add(Dense(1100))
    #model.add(Dropout(0.2))
    #model.add(ELU())
    #model.add(Dense(100))
    #model.add(ELU())
    #model.add(Dense(50))
    #model.add(ELU())
    #model.add(Dense(10))
    #model.add(ELU())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    # Train model on dataset
  
    model.fit_generator(generator = training_generator,
                        validation_data = validation_generator,
                        use_multiprocessing = False,
                        workers = 8,
                        epochs = 5)
    model.save('model.h5')