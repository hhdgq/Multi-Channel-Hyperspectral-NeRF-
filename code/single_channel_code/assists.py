import numpy as np
import random


def generate_training_test(length, train_rate):
    image_list = np.arange(length)
    train_list = sorted(np.random.choice(image_list, int(min(np.ceil(length*train_rate), length)), replace=False))
    val_list = sorted(set(image_list).difference(set(train_list)))
    return train_list, val_list


if __name__ == '__main__':
    pass
