import torch
from absl import logging


def random_vector_with_2norm(_d, norm, dtype=torch.float16, device="cuda"):
    # Generate Uniform Random Variates with Constant Norm
    # https://stats.stackexchange.com/a/487505
    v = torch.normal(0, 1, (_d,), dtype=dtype, device=device)
    v = v * norm / torch.norm(v)
    return v.to(dtype=dtype)


def get_vecs(n, d, miu, sigma, dtype=torch.float16, device="cuda"):
    norms = torch.normal(miu, sigma, (n,))
    V = torch.stack(
        [random_vector_with_2norm(d, e, dtype=dtype, device=device) for e in norms]
    ).to(dtype=dtype, device=device)

    print(V.norm(dim=1))
    print(V.norm(dim=1).mean())
    print(V.norm(dim=1).std())

    V -= V.mean(dim=0)

    V /= V.norm(dim=1).max()
    # V = torch.nn.functional.normalize(V, dim=1)

    logging.info(
        f"Vectors generated with norms mean {V.norm(dim=1).mean():.2f} "
        f"std {V.norm(dim=1).std():.2f}"
    )
    logging.info(f"V centered at {V.mean(dim=0)} with sum {V.mean(dim=0).sum()}")

    return V


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    a = get_vecs(10, 32, 27.7, 23, dtype=torch.float32, device="cuda")
