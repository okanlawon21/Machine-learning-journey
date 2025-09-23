import pandas as pd
from pandas.core.construction import extract_array

data = {"Name": ["Ali", "Mary", "John"], "Score": [85, 90, 78], "CGPA": [4.9, 4.8, 4.88,]}
df = pd.DataFrame(data)
print(df)

import matplotlib.pyplot as plt

x = [85, 90, 78]
y = [4.9, 4.8, 4.88,]

plt.plot(x, y)
plt.title("Simple Graph")
plt.show()

import math

print(int(math.sqrt(400)))     # square root
print(math.sin(math.pi/60)) # sine of 90 degrees



import numpy as np
arr = np.arange(6)
arr_3d = arr.reshape(3,2)
print(arr_3d)

import numpy as np
my_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype = np.uint16)
print("The shape of my array is:", my_array.shape)
print("The number of dimension is:", my_array.ndim)
print("The size of each element in  bytes is:", my_array.itemsize)

import numpy as np
arr = np.ones((3,3), dtype = bool)
print(arr)
arr = np.full((5,5), True, dtype = bool)
print(arr)

import numpy as np
zeros_arr = np.zeros(4)
ones_arr = np.ones(4)
print("Zeros", zeros_arr)
print("ones:", ones_arr)

import numpy as np
arr = np.linspace(5, 50, 10, dtype = int)
print(arr)

import numpy as np
py_list = [1, 2, 3, 4, 5]
arr = np.array(py_list)
print(arr)

import numpy as np
arr = np.arange(10)
nbytes_arr = arr.nbytes
print("Array:", arr)
print("Memory size in bytes:", nbytes_arr)

import numpy as np
arr = np.arange(10)
print(arr[::-1])

import numpy as np
arr_3d = np.eye(3)
print(arr_3d)

import numpy as np

arr = np.arange(1, 17).reshape(4, 4)
First_row = arr[0]
Second_row = arr[1]
last_col = arr[:, -1]
print("Array:\n", arr)
print("First Row:", First_row)
print("Second Row:", Second_row)
print("Last Column:", last_col)

import numpy as np
a=np.array([1, 2, 3])
b=np.array([4, 5, 6])
arr = np.hstack((a,b))
print(arr)


import numpy as np

# Proper array with commas
SampleArray = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12],
    [13,14, 15, 16]
])

# Slice the first 2 rows and first 2 columns
NewArray = SampleArray[0:2, 0:2]

print(NewArray)

import numpy as np
arr= np.arange(1, 17).reshape(4, 4)
print("Array:\n", arr)
sub_arr = arr[0:2, 0:2]
print(sub_arr)

import numpy as np
arr= np.arange(1, 11)
arr[arr % 2 == 1] = -1
print("Original Array:", arr)
print("Modified Array:", arr)

import numpy as np
arr = np.arange(10)
arr[arr % 2 == 0] = -2
print("Original Array:", arr)
print("Modified Array:", arr)

import numpy as array
arr = np.array([1, 0, 2, 0, 3, 0, 4])
Non_zeros = np.nonzero(arr)
print("ARRAY:", arr)
print("Indices", Non_zeros)

import numpy as np
a = np.array([1, 2, 3, 2, 8, 4, 2, 4])
b = np.array([2, 4, 5, 6, 8])
Common_items = np.intersect1d(a, b)
print(Common_items)

import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
work = np.add(a, b)
mult = np.multiply(a, b)
print("Element-wise sum", work)
print("Element-wise multiplication", mult)

import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
prod = np.dot(a, b)
print("Dot Product:", prod)

import numpy as np
arr = np.array([10, 20, 30, 100, 200, 300])
mean = np.mean(arr)
median = np.median(arr)
standard_deviation = np.std(arr)
print("Mean:", int(mean))
print("median", int(median))
print("Standard Deviation:", standard_deviation)

import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])
Non_common = np.logical_and(a,b)
print("Non common elements:", Non_common)

import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])
result = np.setdiff1d(a, b)
print(result)

import numpy as np
arr = np.array([10, 20, 30, 40, 50])
norm_arr = (arr - arr.min()) / (arr.max() - arr.min())
print(norm_arr)

import numpy as np
a= np.array([1, 2, 3, 4, 5])
b = np.array([1, 4,  3, 7, 8])
arr = np.where(a == b,)
print(arr)

import numpy as np
arr = np.arange(15)
ext = arr[(arr >= 5) & (arr <= 10)]
print(ext)

import numpy as np
rand_arr = np.random.rand(3, 2)
print(rand_arr)
MAx = np.max(rand_arr)
print("MAX:", MAx)
MIN = np.min(rand_arr)
print("MIN:", MIN)

import numpy as np
sampleArray = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66] ])
Sorted_indeces = sampleArray[:, 1].argsort()
Sorted_array = sampleArray[Sorted_indeces]
print("Original array:\n", sampleArray)
print("Sorted array:\n", Sorted_array)

import numpy as np
sampleArray = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66] ])
newColumnToAdd = np.array([10, 10, 10])
arr = np.delete(sampleArray, 1, axis = 1)
final_arr = np.insert(arr, 1, newColumnToAdd,axis = 1)
print(final_arr)

import numpy as np
arr = np.arange(9).reshape(3, 3)
print("before")
print(arr)

arr[:, [1, 2]] = arr[:,[2, 1]]
print("after")
print(arr)

import numpy as np
arr = np.random.randint(1, 101, 10)
print(arr)

import numpy as np
rand_arr = np.random.randint(9, size = (3, 3))
print("Original Array:", rand_arr)
sorted_arr = np.sort(rand_arr, axis = 1)
print("Sorted Array:", sorted_arr)


import numpy as np
arr = np.arange(10)
print("Original Array:", arr)
np.random.shuffle(arr)
print("Shuffled Array:", arr)

import numpy as np
arr = np.arange(9).reshape(3, 3)
print("Original Array:", arr)
index = np.argsort(arr, axis = 1)
print("Sorted Array:", index)
