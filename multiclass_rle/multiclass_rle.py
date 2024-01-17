from typing import TypedDict, Annotated, Final

import numpy as np

# Order used by numpy np.ravel (used in the conversion np.array-> RLE) and
# np.reshape (used in the conversion MultiClassRLE->np.array) methods.
# From numpy's doc:
#    """
#    ‘F’ means to index the elements in column-major, Fortran-style order,
#    with the first index changing fastest, and the last index changing slowest
#    """
# This is what pycocotools uses
NUMPY_ORDER_RAVEL: Final = "F"
NUMPY_ORDER_RESHAPE: Final = "F"


class MultiClassRLE(TypedDict):
    size: Annotated[tuple[int, int], "(H, W)"]
    counts: Annotated[tuple[int, ...], "N_pixels == H*W"]
    values: Annotated[tuple[int, ...], "N_pixels == H*W"]


MaskArray = Annotated[np.ndarray, "(H, W)"]


def multi_class_rle_to_array(multi_class_rle: MultiClassRLE) -> MaskArray:
    """Convert a multi class RLE to a multiclass 2D arrays.

    Args:
        multi_class_rle (MultiClassRLE): multi class RLE

    Returns:
        np.ndarray: corresponding multiclass 2D arrays
    """
    height, witdh = multi_class_rle["size"]
    mask_array = np.zeros((height * witdh), dtype=np.uint8)
    start = 0
    for val, cnt in zip(multi_class_rle["values"], multi_class_rle["counts"]):
        mask_array[start : start + cnt] = val
        start += cnt
    return mask_array.reshape((height, witdh), order=NUMPY_ORDER_RESHAPE)


def array_to_multi_class_rle(mask_array: MaskArray) -> MultiClassRLE:
    """Convert a multiclass 2D arrays to a multi class RLE.

    Args:
        mask_array (MaskArray): multiclass 2D arrays

    Returns:
        MultiClassRLE:  multi class RLE
    """
    H, W = mask_array.shape
    mask_array = mask_array.ravel(order=NUMPY_ORDER_RAVEL)
    pad_array = mask_array
    pad_array = np.append([pad_array[0] + 1], pad_array)
    pad_array = np.append(pad_array, [pad_array[-1] + 1])
    start = np.where(pad_array[1:] != pad_array[:-1])[0]
    return {
        "size": (H, W),
        "values": tuple(mask_array[start[:-1]].tolist()),
        "counts": tuple((start[1:] - start[:-1]).tolist()),
    }
