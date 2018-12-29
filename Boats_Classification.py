import os
from numpy.random import seed
from tensorflow import set_random_seed

from dir_vars import directory_vars
from data_processing import generate_augment_data
from model_handling import (
    t_model_vary_lr, fit_model_varying_lr,
    plot_model_acc, plot_model_loss, eval_predict_model)

# Disable tensorflow gpu basic log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# To ensure reproducibility
seed(1)
set_random_seed(2)

# Setting the environment variables
training_dir, testing_dir, ground_truth_dir = directory_vars()

# Generate augmented data from data generators
# View an image of the data generator
# num_imgs must be divisible by 3
(train_generator, validation_generator,
 test_generator, y_true, num_classes) = generate_augment_data(training_dir,
                                                              testing_dir,
                                                              ground_truth_dir,
                                                              view_img=True,
                                                              num_imgs=12)

# Init the model
model = t_model_vary_lr(
    image_size=256, num_classes=num_classes,
    view_model=True, weights_path=None)

# Fit the model
model_history = fit_model_varying_lr(
    model, train_generator, validation_generator)

# Plot model history
# summarize history for accuracy (acc, val_acc)
plot_model_acc(model_history)
# summarize history for loss (loss, val_loss)
plot_model_loss(model_history)

# Test the model
predictions, scores = eval_predict_model(
    model, test_generator, y_true, save_model=True)
