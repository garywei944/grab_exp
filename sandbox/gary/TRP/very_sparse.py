import numpy as np

from sklearn.utils.random import sample_without_replacement
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils import check_random_state

import torch_sparse
from absl import logging

import torch

from grablib.utils.random_projection import *

logging.set_verbosity(logging.INFO)

dtype = torch.float32
# dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = int(117e6)  # gpt2-small
# d = int(762e6)  # gpt2-medium
# d = int(69e9)
b = 2
n = 300e3

X = torch.rand(
    b,
    d,
    dtype=dtype,
    device=device,
)
#
# # print(X.permute(*torch.arange(X.ndim - 1, -1, -1)).shape)
# # r = torch.einsum("ik,...k->...i", PI, X)
#
# # print(r.shape)
# # print(r.mean())
# # print(r.max())
# # print(r.min())
#
# # print(_sparse_random_matrix(1000, d, density=0.06, random_state=42))
#
# # t = SparseTensor.from_torch_sparse_csr_tensor(PI)
#
# # r = torch_sparse.matmul(PI, X.reshape(-1, 1)).squeeze(-1)
#
# # r = PI.matmul(X.reshape(-1, 1)).squeeze(-1)
#
# logging.set_verbosity(logging.INFO)
#
# pi = JLVerySparseRP(n=2000, d=d, dtype=dtype, device=device)
# r = pi.project(X)


pi = KroneckerRP(n=n, d=d, dtype=dtype, device=device, order=3, sparse=False)
r = pi.project(X)

print(r)
print(X.norm(dim=-1))
print(r.norm(dim=-1))
print(r.shape)
