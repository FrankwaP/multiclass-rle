# multiclass_rle

## How to use multiclass_rle

[Run-length Encoding](https://en.wikipedia.org/wiki/Run-length_encoding) (RLE) for multiclass 2D mask arrays (e.g. image segmentation targets).  

It is built as an extension of the COCO format, by simply adding a 'values' key to the dictionnary used.  

That way, a 2D array such as:

```python
array = np.array([
    [0, 0, 1, 1, 0],
    [3, 0, 1, 1, 0],
    [3, 3, 1, 1, 0],
    [3, 3, 0, 0, 0]
])
```

… can be encoded as:

```python
rle = {
    'size': (4, 5),
    'values': (0, 3, 0, 3, 1, 0, 1, 0)
    'counts': (1, 3, 2, 2, 3, 1, 3, 5)
}
```

(Note that, like COCO format, it uses the same "rows first", or Fortran, convention.)

It is not visible on the example above since the array is very small, but RLE allows to drastically reduce the memory needed to store a mask array.  
For performance purpose, this library uses as much `numpy` as possible. And it pays off, as shown in [benchmark](benchmark.ipynb)!  

## Installation

You can freely copy/paste the code from multiclass_rle/multiclass_rle.py

Else, if you want to install multiclass_rle from GitHub repository, do:

```console
git clone https://github.com/FrankwaP/multiclass-rle.git
cd multiclass-rle
python -m pip install .
# or on mac: python3 -m pip install .
```

## Documentation

### Encoding

When you're preparing your data and want to store them in a memory-efficient way, use array_to_multi_class_rle:  

```python
import numpy as np

from multiclass_rle import array_to_multi_class_rle

array = np.array([
    [0,0,1,1,0],
    [3,0,1,1,0],
    [3,3,1,1,0],
    [3,3,0,0,0]
])

rle = array_to_multi_class_rle(array)
print(rle)
```

Note that the RLE object are built on Python's tuple and dictionary objects, so they are easily storable in a json of pickle object for example.

### Decoding

To decode them so you can work with the numpy array, use `multi_class_rle_to_array`.  

```python
from multiclass_rle import multi_class_rle_to_array

rle = {
    'size': (4, 5),
    'values': (0, 3, 0, 3, 1, 0, 1, 0),
    'counts': (1, 3, 2, 2, 3, 1, 3, 5)
}

arr = multi_class_rle_to_array(rle)
print(arr)
```

Here's how it looks like when one wants to train a multiclass semantic segmentation model:

```python

from multiclass_rle import multi_class_rle_to_array

# ...

dataset = load_dataset(path, ...)
batch_generator = BatchGenerator(dataset, ...)

for epoch in range(N_epoch):
    # ...
    for batch in batch_generator:
        # ...
        for image, mask in batch:  # 
            bin_mask = multi_class_rle_to_array(mask)
            # ...
        # ...
    # ...
# ...
```
