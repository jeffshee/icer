import torch

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import dlib


def test_env():
    """
    CUDA testing
    """
    print("PyTorch", torch.cuda.is_available())
    print("Tensorflow", tf.test.is_built_with_cuda(), tf.config.list_physical_devices("GPU"))
    print("Dlib", dlib.DLIB_USE_CUDA, dlib.cuda.get_num_devices())


if __name__ == "__main__":
    test_env()
