import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def preprocess_images():
    '''
    Training Data generator, rescales images to 1.0/255,
    apply both shear_range, zoom_range with value of 0.2,
    horizontal_flip & vertical_flip.

    Testing Data generator, only rescales the images

    Returns:
        train_datagen -- ImageDataGenerator,
        test_datagen -- ImageDataGenerator
    '''

    # The augmentation for training set
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True
                                       )
    # The augmentation for testing set
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen, test_datagen


def prepare_data(batch_size, train_datagen, test_datagen, training_dir,
                 testing_dir, ground_truth_dir, visualize):

    # Generate batches of image data (and their labels)
    #  directly from training folders batch_size

    # The generator will read pictures found in
    # subfolders of training_data & generate batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        training_dir, batch_size=batch_size, class_mode='categorical'
    )

    # Further insights about our testing data-set
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
    test_generator = test_datagen.flow_from_dataframe(new_df, testing_dir)

    # Visualize our data sets
    if(visualize):
        train_samples = train_generator.samples
        train_num_classes = train_generator.num_classes

        testing_samples = len(new_df.values)
        unique_testing_classes = new_df.groupby('class')['class'].nunique()
        testing_classes_num = len(unique_testing_classes.values)

        print('Training set: ({}, {})'.format(
            train_samples, train_num_classes))
        print('Testing set: ({}, {})'.format(
            testing_samples, testing_classes_num))

    return train_generator, test_generator, train_num_classes
