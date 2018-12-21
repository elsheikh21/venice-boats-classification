import os


def directory_vars():
    training_dir = os.path.join(
        os.getcwd(), 'training_set')
    testing_dir = os.path.join(os.getcwd(), 'testing_set')
    ground_truth_dir = os.path.join(testing_dir, 'test', 'ground_truth.txt')
    return training_dir, testing_dir, ground_truth_dir
