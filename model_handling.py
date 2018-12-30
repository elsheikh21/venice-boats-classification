import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.python.keras.callbacks import LearningRateScheduler
import math
from keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import (
    Flatten, BatchNormalization, Dropout)

# Tune out the learning rate
num_epochs = 100
init_lr = 1e-6


def integer_encoding(data):
    encoder = LabelEncoder()
    return encoder.fit_transform(data)


def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    max_epochs = num_epochs
    base_lr = init_lr
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = base_lr * (1 - (epoch / float(max_epochs))) ** power

    # return the new learning rate
    return alpha


def lenet_model_vary_lr(image_size, num_classes, view_model=True):
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(20, (5, 5), input_shape=(
        image_size, image_size, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    opt = SGD(lr=init_lr, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    if(view_model):
        print(model.summary())

    return model


def t_model_vary_lr(image_size, num_classes, view_model=True,
                    weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 5), input_shape=(
        image_size, image_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))  # 0.5 good value
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = SGD(lr=init_lr, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    if weights_path is not None:
        model.load_weights(weights_path)

    if(view_model):
        print(model.summary())

    return model


def ta_model_vary_lr(image_size, num_classes, view_model=True,
                     weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 5), input_shape=(
        image_size, image_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = SGD(lr=init_lr, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    if weights_path is not None:
        model.load_weights(weights_path)

    if(view_model):
        print(model.summary())

    return model


def fit_model_varying_lr(model, train_generator, validation_generator):
    early_stopping = EarlyStopping(
        monitor='acc', patience=5, verbose=1, mode='max')
    csv_logger = CSVLogger('model_training_log.log')
    callbacks = [LearningRateScheduler(poly_decay), csv_logger, early_stopping]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(
            train_generator.n//train_generator.batch_size),
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=math.ceil(validation_generator.n //
                                   validation_generator.batch_size),
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=0,
        shuffle=False
    )

    return history


def plot_model_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


def plot_model_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


def eval_predict_model(model, test_generator, y_true,
                       save_model=True, save_name='trail'):
    predictions = model.predict_generator(
        test_generator, test_generator.n // test_generator.batch_size,
        verbose=1
    )

    y_pred = np.argmax(predictions, axis=1)

    acc = round(accuracy_score(y_true, y_pred) * 100.0, 2)
    print('\nThe Accuracy score is: {} % \n'.format(str(acc)))
    print('\nThe Classification Report: \n',
          classification_report(y_true, y_pred))

    scores = model.evaluate_generator(
        test_generator, test_generator.n // test_generator.batch_size,
        verbose=1
    )

    if(save_model):
        model.save('{} model.h5'.format(save_name))
        model.save_weights('{} model_weights.h5'.format(save_name))

    return predictions, scores
