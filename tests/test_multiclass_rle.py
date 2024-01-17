"""Tests for the multiclass_rle.mutliclass_rle module.
"""
import pytest
import numpy as np

from multiclass_rle.multiclass_rle import multi_class_rle_to_array, array_to_multi_class_rle


arr0 = np.array([[0, 1, 2], [0, 1, 2]])
rle0 = {"size": (2, 3), "values": (0, 1, 2), "counts": (2, 2, 2)}


arr1 = np.array([[0, 0], [1, 1], [2, 2]])
rle1 = {"size": (3, 2), "values": (0, 1, 2, 0, 1, 2), "counts": (1, 1, 1, 1, 1, 1)}


all_arr = [arr0, arr1]
all_rle = [rle0, rle1]


def test_multi_class_rle_to_array():
    for arr, rle in zip(all_arr, all_rle):
        assert np.all(multi_class_rle_to_array(rle) == arr)


def test_array_to_multi_class_rle():
    for arr, rle in zip(all_arr, all_rle):
        assert array_to_multi_class_rle(arr) == rle
