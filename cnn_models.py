import pandas as pd
from numpy import savetxt
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten


def alexnet_model(img_shape, n_classes, l2_reg,
                  weights_path=None, visualize_summary=False):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
                       padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights_path is not None:
        alexnet.load_weights(weights_path)

    alexnet.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    if(visualize_summary):
        print('\nAlexNet Model:')
        print(alexnet.summary())

    return alexnet


def fit_save_alexnet_model(alexnet_model, train_generator, save_model=True):
    # Tracking the models
    csv_logger2 = CSVLogger('alexnet_model_training.log')

    print('\nFitting the model process has begun...\n')
    alexnet_model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.n//train_generator.batch_size),
        epochs=50,
        verbose=1,
        callbacks=[csv_logger2],
        use_multiprocessing=True,
        workers=0
    )
    print('\nAlexNet Model fitting process has ended successfully.')
    print("Log file is generated with the name 'alexnet_model_training.log'")

    if(save_model):
        # Save model weights
        alexnet_model.save_weights('alexnet_model_weights.h5')
        # Save model for further analysis
        alexnet_model.save("alexnet_model.h5")

    alexnet_log_df = pd.read_csv('alexnet_model_training.log', delimiter=',')
    alexnet_log_df.to_csv('alexnet_model_training.csv')
    print("Log file is generated with the name 'alexnet_model_training.csv'")


def predict_eval_alexnet_model(alexnet_model, test_generator):
    print('\nAlexNet Model evaluation & prediction process starting...')
    try:
        predict = alexnet_model.predict_generator(
            test_generator, test_generator.n // test_generator.batch_size,
            verbose=1)
        scores = alexnet_model.evaluate_generator(
            test_generator, test_generator.n // test_generator.batch_size,
            verbose=1)
        savetxt('AlexNet_scores.txt', scores)
        savetxt('AlexNet_predictions.txt', predict)
        print('Evaluation and prediction scores are saved.')
    except BaseException as error:
        print('An exception occurred: {}'.format(error))


def lenet_model(img_shape, train_classes,
                weights_path=None, visualize_summary=False):
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(20, (5, 5), input_shape=img_shape))
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
    model.add(Dense(train_classes))
    model.add(Activation("softmax"))

    # if weightsPath is specified load the weights
    if weights_path is not None:
        model.load_weights(weights_path)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    if(visualize_summary):
        print('\nLeNet Model:')
        print(model.summary())

    return model


def fit_save_lenet_model(lenet_model, train_generator, save_model=True):
    # Tracking our model
    csv_logger1 = CSVLogger('lenet_model_training.log')
    # Fit generated data to the model
    print('\nFitting the model process has begun...\n')
    lenet_model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.n//train_generator.batch_size),
        epochs=50,
        verbose=1,
        callbacks=[csv_logger1],
        use_multiprocessing=True,
        workers=0
    )
    print('\nLeNet Model fitting process has ended successfully.')
    print("Log file is generated with the name 'lenet_model_training.log'")

    if(save_model):
        # Save model weights
        lenet_model.save_weights('lenet_model_weights.h5')
        # Save model for further analysis
        lenet_model.save("lenet_model.h5")

    # for better visualizing our model log
    lenet_log_df = pd.read_csv('lenet_model_training.log', delimiter=',')
    lenet_log_df.to_csv('lenet_model_training.csv')
    print("Log file is generated with the name 'lenet_model_training.csv'")


def predict_eval_lenet_model(lenet_model, test_generator):
    print('LeNet Model evaluation & prediction process starting...')
    try:
        predict = lenet_model.predict_generator(
            test_generator, test_generator.n // test_generator.batch_size,
            verbose=1)
        scores = lenet_model.evaluate_generator(
            test_generator, test_generator.n // test_generator.batch_size,
            verbose=1)
        savetxt('LeNet_scores.txt', scores)
        savetxt('LeNet_predictions.txt', predict)
        print('Evaluation and prediction scores are saved.')
    except BaseException as error:
        print('An exception occurred: {}'.format(error))


def vgg_16(img_size, weights_path=None, visualize_summary=True):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=img_size))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    if(visualize_summary):
        print('\nVGG16 Model:')
        print(model.summary())

    return model


def fit_save_vgg_16_model(vgg16_model, train_generator, save_model=True):
    # Tracking our model
    csv_logger = CSVLogger('VGG16_model_training.log')
    # Fit generated data to the model
    print('\nFitting the model process has begun...\n')
    vgg16_model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.n//train_generator.batch_size),
        epochs=50,
        verbose=1,
        callbacks=[csv_logger],
        use_multiprocessing=True,
        workers=0
    )

    if(save_model):
        vgg16_model.save('VGG16_model.h5')
        vgg16_model.save_weights('VGG16_model_weights.h5')


def predict_eval_vgg_16_model(vgg16_model, test_generator):
    print('VGG16 Model evaluation & prediction process starting...')
    try:
        predict = lenet_model.predict_generator(
            test_generator, test_generator.n // test_generator.batch_size,
            verbose=1)
        scores = lenet_model.evaluate_generator(
            test_generator, test_generator.n // test_generator.batch_size,
            verbose=1)
        savetxt('VGG16_scores.txt', scores)
        savetxt('VGG16_predictions.txt', predict)
        print('Evaluation and prediction scores are saved.')
    except BaseException as error:
        print('An exception occurred: {}'.format(error))
