from torch import allclose, cuda, device, manual_seed
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential

from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad
from backpack.utils.examples import load_one_batch_mnist

# make deterministic
manual_seed(0)

dev = device("cuda" if cuda.is_available() else "cpu")

# data
X, y = load_one_batch_mnist(batch_size=128)
X, y = X.to(dev), y.to(dev)

# model
model = Sequential(Flatten(), Linear(784, 10)).to(dev)
lossfunc = CrossEntropyLoss().to(dev)

model = extend(model)
lossfunc = extend(lossfunc)

# selected samples
subsampling = [0, 1, 13, 42]

loss = lossfunc(model(X), y)

with backpack(BatchL2Grad()):
    loss.backward()

# naive approach: compute for all, slice out relevant
# naive = [p.grad_batch[subsampling] for p in model.parameters()]

print('')
