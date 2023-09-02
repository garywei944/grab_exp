# Quick Usage for GraB-lib

In short, the GraB library is just a PyTorch dataloader sampler with an internal
sorter.
The sorter implements the GraB algorithm and the sampler is a PyTorch wrapper.

## Use GraB PyTorch sampler

For basic usages, check out [the pypi package](https://test.pypi.org/project/grabngo/) 
for a usage example.

## Directly use the Sorter

Just in case you wanna directly use the GraB algorithm, create the corresponding
sorter class, e.g. `MeanBalance`.

```python
from grablib.sorter import MeanBalance

n = 1000
d = 200

sorter = MeanBalance(n, d)
```

`MeanBalance` inherits `SorterBace` under `grabngo/sorter/SorterBase`. 
`SorterBase` defines other arguments like `record_herding`, `random_projection`, etc,
you are free to check them out, but for minimum usage, they are not necessary.

Only 2 methods of SorterBase is necessary for minimum use,

```python
sorter.step(per_sample_grads: dict[str, Tensor])
# the per_sample_grads follows the same structure of PyTorch model.named_parameters()

# Let's say you have an arbitrary n*d data tensor
demo_data = torch.rand(n, d)

# 1 step of GraB update is very simple
sorter.step({'': demo_data})

# When you finish the full epoch, simply call reset_epoch() to get ready for the
# next epoch
sorter.reset_epoch()
# reset the variables used by GraB and get the orders ready for the next epoch
```

## Notes for record herding and average gradient error

I currently implements recording herding bound and average gradient error in the
GraB-lib instead of training scripts for code reuse.

I tried to make the project more "software engineering" with clean structure and
straightforward method and variable names.
Let me konw if any codes are confusing!
