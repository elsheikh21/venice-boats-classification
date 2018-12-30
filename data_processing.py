import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_handling import integer_encoding
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 256


def generate_augment_data(training_dir, testing_dir, ground_truth_dir,
                          view_img=True, num_imgs=12):
    #   To ensure consistency for our experiments
    x = np.random.seed(7)
    data_generator = ImageDataGenerator(
        rotation_range=360,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )

    batch_size = 32

    train_generator = data_generator.flow_from_directory(
        training_dir,
        batch_size=batch_size,
        subset='training',
        shuffle=False,
        target_size=(image_size, image_size),
        seed=x
    )

    num_classes = len(train_generator.class_indices)

    test_datagen = ImageDataGenerator()

    validation_generator = data_generator.flow_from_directory(
        training_dir,
        batch_size=batch_size,
        subset='validation',
        shuffle=False,
        target_size=(image_size, image_size),
        seed=x
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

    if(view_img):
        boat_image_path = os.path.join(
            training_dir, 'Polizia', '20130304_061409_08018.jpg')
        image = np.expand_dims(plt.imread(boat_image_path), 0)
        plt.imshow(image[0])
        # Generate batches of augmented image from original image
        aug_iter = data_generator.flow(image)

        # Get 12 samples of our augmented image
        aug_images = [next(aug_iter)[0].astype(np.uint8)
                      for i in range(num_imgs)]
        plots(aug_images, figsize=(80, 40), rows=6)

    return (train_generator, validation_generator,
            test_generator, y_true, num_classes)


# Plot an image
def plots(ims, figsize=(24, 12), rows=2, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()
