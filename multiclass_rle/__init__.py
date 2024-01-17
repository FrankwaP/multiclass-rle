"""Documentation about multiclass_rle"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "François P."
__email__ = "francois.plessier@gmail.com"
__version__ = "0.1.0"

from .multiclass_rle import multi_class_rle_to_array, array_to_multi_class_rle
