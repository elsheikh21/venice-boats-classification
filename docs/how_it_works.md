# How it works

1. Open `boats_classification.py` and run it

   1. We first bump into `directory_vars()`

      - Using python native `os` library, we just set the path for training set, testing set, and the ground truth file.
      - `dir_vars.directory_vars()` returns 3 variables, training_dir, testing_dir, ground_truth_dir
        - But **Why??** Just following the academic paradigm of coding and to increase my code readability.

   2. After reading the paths, we need them to read in our dataset from the previously specified locations using `generate_augment_data()`

      i. Using a `np.random.seed` to ensure reproducibility of results; as, we will have more than a model to try to conclude few things at last.

      ii. Having 2 data generators, one used for training data, the other is for testing data.

      - But **WHY??** One way to fight overfitting and increase the overall performance of our classifier, as well as, trying to avoid an unbalanced dataset, is to generate different representations of our object, so that the classifier gets to know more about the object and thus extracting the most valuable features.
        - So we flipped the images horizontally and vertically, rotated the images, as well as, applying zoom_range, height & width_shift_range, and, shear_range, finally, we split it into (20%) validation and (80%) training data using `ImageDataGenerator`.
          - validation set is to further tune the parameters of a classifier
        - But for testing, I just left it as it is; as we are testing the classifier performance on those given images and no need for altering them.
      - After that, we just allow the `ImageDataGenerator` to flow our samples of data using `flow_from_directory`

      iii. I defined a batch size of 32 samples, to be fed through the network

      iv. So, now we are done with both, training and validation data, let's turn our heads to the testing data

      - First, we read the ground truth file using `pd.read_csv()`, and split data into classes, and numbers of samples per class. And, we get the encodings for those classes using `integer_encoding`. And, not to forget that we need to remove those classes we mentioned in the dataset section earlier.
      - Secondly, we just need to flow data from the prespecified path using `flow_from_dataframe`, dataframe?? this is what `pd.read_csv()` returns, right? Yes, right.
      - Finally, plotting the results of augmentation, because, what is machine learning without visualizing your inputs, outputs.... It is way more than that but it is an important thing to do.

      v. So, to wrap up....

      - We flow data from the training directory variable passed to the method when invoked, to augment data -to combat overfitting-, split them images into training and validation sets.
      - Use the ground truth variable to build from it the dataframe to get the encodings of each class to be able to calculate the overall classifier's performance.
      - Load the testing set using the testing directory variable passed to the method.
      - and the last 2 parameters are view_images flag, and number of images to be shown, again, to visualize our data

   3. To the real deal, I am using keras on top of a tensorflow as backend, why? due to the large community and this is hugely important. Apart from that, I am using tensorflow gpu, because it is fast.

      - Used keras, for its simplicity, to build the model consisting of 3 convolutional layers and an output layer
        - Since, this is a multi class classification problem, we used softmax and ReLu activation functions, categorical cross entropy as the loss function.
        - Let's get back to our model, as I mentioned it was built using 3 convolution layers (Conv2D layer -> ReLu Activation -> MaxPooling2D) and a (ZeroPooling2D) layer in between of both convolution layers, why? Because the layer progressively reduces the spatial size of the representation to reduce the number of parameters and computation in the network, and hence to also control overfitting.
        - Batch Normalization layer was added between the Conv2D layer and the 3rd ReLu layer, as it normalizes the output of the Conv2D layer to the ReLu.
        - Followed by Flatten layer was used to turn the feature vector into a 1D to be used by the ANN classifier layer.
        - Finally, before the output layer we have a Dropout layer with probability of 0.7 to switch off some neurons of our NN, to prevent overfitting.
        - The output layer, having input as number of our classes, which we computed in the second step of this algorithm.
        - But, what about other models you used, actually you can find a full report ([here](Report.pdf)), discussing everything. Please make yourself comfortable.
      - I tweaked several params of the fitting process, number of epochs was set to 100 to make sure that the classifier takes its time to train, however, this might cause overfitting, so how to combat that? Early Stopper, my friend, so I used Keras APIs to have an early stop if there is no improvement over 5 epochs in maximizing the acc, why not any other param, actually, based on trail and error, this one yielded the best results.
      - But what about steps per epoch & validation steps, actually, I followed a convention of number of samples we have over the batch size.
      - Callbacks, these saviors are invoked on epoch end, added to them the early stopping criteria I was talking about, as well as, a logger, just to log the epochs, yes, simply that, and finally, the learning rate tuner. Wait, what? Let's take it one step at a time, this thing over here all it does, after every epoch it tries to find the best/optimum learning rate, to help our optimizer to reach the global minima, or maxima, depends upon how you visualize it, just simple tweak but saves a lot of time.
      - Optimizer? Yes, this was chosen to be Stochastic Gradient Descent (SGD) with Nesterov Momentum, which is famous for faster convergence than the regular SGD.
      - Lastly, since we are using GPU, why not to have more performance advantage in terms of computations and roast our gpu by using multi processing, as well as, running it on our PC's main thread.

   4. Back to visualization, after fitting our data to the classifier, we need to check plots for loss vs val_loss, acc vs val_acc. But what's the difference? simply loss function output is for the training data, val_loss is for validation data loss value, same applies for acc & val_acc. Why the plots? to visualize and get further insights of the model performance over the number of epochs. So, we used `plot_model_acc` and `plot_model_loss`

   5. Finally, we need to test our classifier and how good was it? So, we invoked the method `eval_predict_model` passing the model, the testing generator we used earlier, y_true, save_model flag, and of course the saving name of it. It prints out the classification report, as well as, the overall classifier accuracy.
      - The best accuracy the model got was of 61.34%, which is very good, at least from my point of view.
