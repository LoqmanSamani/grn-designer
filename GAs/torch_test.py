import torch
import time
import numba
import numpy

x = torch.tensor([i for i in range(1000000)])
y = torch.tensor([i for i in range(1000000)])

x1 = numpy.array([i for i in range(1000000)])
y1 = numpy.array([i for i in range(1000000)])

@torch.jit.script
def vec_add_odd_pos(a, b):
    res = 0.
    for pos in range(len(a)):
        if pos % 2 == 0:
            res += a[pos] + b[pos]
    return res

def vec_add_odd_pos1(a, b):
    res = 0.
    for pos in range(len(a)):
        if pos % 2 == 0:
            res += a[pos] + b[pos]
    return res

@numba.jit(nopython=True)
def vec_add_odd_pos2(a, b):
    res = 0.0
    for pos in range(len(a)):
        if pos % 2 == 0:
            res += a[pos] + b[pos]
    return res



# Timing TorchScript function
tic = time.time()
z = vec_add_odd_pos(x, y)
toc = time.time()
t = toc - tic
print(f"TorchScript time: {t}")


# Timing plain Python function
tic1 = time.time()
z1 = vec_add_odd_pos1(x, y)
toc1 = time.time()
t1 = toc1 - tic1
print(f"Plain Python time: {t1}")

# Timing Numba function
tic2 = time.time()
z2 = vec_add_odd_pos2(x1, y1)
toc2 = time.time()
t2 = toc2 - tic2
print(f"Numba time: {t2}")




# Timing Numba function (convert tensor to ndarray)
tic3 = time.time()
x_np = x.numpy()
y_np = y.numpy()
z3 = vec_add_odd_pos2(x_np, y_np)
toc3 = time.time()
t3 = toc3 - tic3
print(f"Numba time(tensor to ndarray): {t3}")

# TorchScript time: 12.033197402954102
# Plain Python time: 8.29473328590393
# Numba time: 0.5144443511962891
# Numba time(tensor to ndarray): 0.09679460525512695


tic4 = time.time()
for i in range(20):
    z2 = vec_add_odd_pos2(x1, y1)
toc4 = time.time()
t4 = toc4 - tic4


tic5 = time.time()
for i in range(20):

    x_np = x.numpy()
    y_np = y.numpy()
    z = vec_add_odd_pos2(x_np, y_np)

toc5 = time.time()
t5 = toc5 - tic5

print(t4)
print(t5)
# 0.02430582046508789
# 0.023659706115722656

