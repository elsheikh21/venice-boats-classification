import os
from dir_vars import directory_vars
from data_processing import generate_augment_data
from model_handling import (
    t_model_vary_lr, fit_model_varying_lr,
    plot_model_acc, plot_model_loss, eval_predict_model,
    lenet_model_vary_lr, ta_model_vary_lr)

# Disable tensorflow gpu log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Init the model A
modelA = t_model_vary_lr(
    image_size=256, num_classes=num_classes,
    view_model=True, weights_path=None)

# Init model A with more dropout layers
modelA_i = ta_model_vary_lr(
    image_size=256, num_classes=num_classes,
    view_model=True, weights_path=None)

# Init model B (LeNet)
modelB = lenet_model_vary_lr(
    image_size=256, num_classes=num_classes,
    view_model=True)

# Fit the models
modelA_history = fit_model_varying_lr(
    modelA, train_generator, validation_generator)

modelA_i_history = fit_model_varying_lr(
    modelA_i, train_generator, validation_generator)

modelB_history = fit_model_varying_lr(
    modelB, train_generator, validation_generator)

# Plot models history

plot_model_acc(modelA_history)
plot_model_loss(modelA_history)

plot_model_acc(modelA_i_history)
plot_model_loss(modelA_i_history)

plot_model_acc(modelB_history)
plot_model_loss(modelB_history)

# Test the models
predictions, scores = eval_predict_model(
    modelA, test_generator, y_true, save_model=True, save_name='Model A')

predictions, scores = eval_predict_model(
    modelA_i, test_generator, y_true, save_model=True,
    save_name='Model A variance I')

predictions, scores = eval_predict_model(
    modelB, test_generator, y_true, save_model=True, save_name='LeNet')
