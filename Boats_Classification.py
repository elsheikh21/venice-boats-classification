from set_directory_variables import directory_vars
from data_handling import (prepare_data, preprocess_images)
from cnn_models import (lenet_model, fit_save_lenet_model,
                        predict_eval_lenet_model, vgg_16,
                        fit_save_vgg_16_model,
                        predict_eval_vgg_16_model)

# To use the alexnet models, just uncomment this import
# from cnn_models import (alexnet_model, fit_save_alexnet_model,
# predict_eval_alexnet_model)

# Step 1: set our directory variables
training_dir, testing_dir, ground_truth_dir = directory_vars()

#  Step2: Data preprocessing & augmentation
train_datagen, test_datagen = preprocess_images()

# Step3: Prepare the data
train_generator, test_generator, train_num_classes = prepare_data(
    32, train_datagen, test_datagen, training_dir,
    testing_dir, ground_truth_dir, True)

# Step4: Training Our model
# LeNet Model
lenet_model = lenet_model(img_shape=(256, 256, 3),
                          train_classes=train_num_classes,
                          weights_path=None, visualize_summary=True)

fit_save_lenet_model(lenet_model, train_generator, save_model=True)

# AlexNet Model
# alexnet_model = alexnet_model(img_shape=(256, 256, 3),
#                               n_classes=train_num_classes, l2_reg=0.0,
#                               weights=None, visualize_summary=True)

# fit_save_alexnet_model(alexnet_model, train_generator, save_model=True)

# VGG16 Model
vgg16_model = vgg_16(img_size=(256, 256, 3),
                     weights_path=None, visualize_summary=True)

fit_save_vgg_16_model(lenet_model, train_generator, save_model=True)

# Step5: Test our model
test_lenet_model = predict_eval_lenet_model(lenet_model, test_generator)

# test_alexnet_model = predict_eval_alexnet_model(alexnet_model, test_generator)

test_vgg16_model = predict_eval_vgg_16_model(vgg16_model, test_generator)
