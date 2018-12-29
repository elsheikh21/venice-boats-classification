import pandas as pd
from model_handling import integer_encoding
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 256


def generate_augment_data(training_dir, testing_dir, ground_truth_dir):
    data_generator = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        validation_split=0.2
    )

    batch_size = 32

    train_generator = data_generator.flow_from_directory(
        training_dir,
        batch_size=batch_size,
        subset='training',
        shuffle=False,
        target_size=(image_size, image_size)
    )

    num_classes = len(train_generator.class_indices)

    test_datagen = ImageDataGenerator()

    validation_generator = data_generator.flow_from_directory(
        training_dir,
        batch_size=batch_size,
        subset='validation',
        shuffle=False,
        target_size=(image_size, image_size)
    )

    df = pd.read_csv(ground_truth_dir, delimiter=';')
    df.columns = ['filename', 'class']

    # Remove other data not in training set
    to_remove_list = ['Cacciapesca', 'Caorlina',
                      'Lanciamaggioredi10mMarrone', 'Sanpierota',
                      'VigilidelFuoco',
                      'SnapshotBarcaParziale', 'SnapshotBarcaMultipla',
                      'Mototopocorto']

    # new dataframe to be used to process the data
    new_df = df[~df['class'].isin(to_remove_list)]

    # Similar generator, for testing data
    test_generator = test_datagen.flow_from_dataframe(
        new_df,
        testing_dir,
        batch_size=1,
        shuffle=False,
        target_size=(image_size, image_size)
    )

    y_true = integer_encoding(new_df['class'])

    return (train_generator, validation_generator,
            test_generator, y_true, num_classes)
