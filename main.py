import functools
import time

import numpy as np

from model import gpt2, softmax
from utils import load_encoder_hparams_and_params


def max_fn(x):
    x_max = np.where(x > 0, x, 0)
    return x_max / np.sum(x_max)


def sample(probs):
    # greedy decode, you could also sample from the distribution directly of course
    return np.argmax(probs)


def autoregressive_sampling(x, model, N):
    n = len(x)
    T = len(x) + N

    while n < T:
        q = model(x)
        next_id = sample(q[-1])
        x = np.append(x, next_id)
        n += 1

    return x


def speculative_sampling(x, draft_model, target_model, N, K):
    # NOTE: paper indexes arrays starting from 1, python indexes from 0, so
    # we have to add an extra -1 term when indexing using n, T, or t
    n = len(x)
    T = len(x) + N

    while n < T:
        # Step 1: auto-regressive decode K tokens from draft model
        for _ in range(K):
            x = np.append(x, sample(draft_model(x)[-1]))

        # Step 2: draft and target model forward passes
        p = draft_model(x)
        q = target_model(x)

        # Step 3: figure out how many of the draft predictions we want to keep based on
        # the rejection criteria, which is stored in the variable t
        for t in range(1, K + 1):
            r = np.random.random()
            i = n + t - 2
            j = np.argmax(p[i + 1])
            if not r < min(1, q[i][j] / p[i][j]):
                break

        # Step 4: sample final token
        if t == K:
            next_id = sample(q[-1])
            x = np.append(x, next_id)
            n += 1
        else:
            x = x[: -K + t]  # only keep accepted draft tokens
            if t == 0:
                next_id = sample(max_fn(q[n - 1] - p[n - 1]))  # resample
                x = np.append(x, next_id)
                n += 1

        # update n by the total number of tokens we decoded
        n += t
        assert n == len(x)  # just keeping my sanity

    return x


def create_model_fn(params, hparams):
    f = functools.partial(gpt2, **params, n_head=hparams["n_head"])
    g = lambda inputs: softmax(f(inputs))
    # NOTE: if you want to implement top-p/top-k etc ..., you need to modify the
    # probability distribution here instead of in the sample function, since in the
    # paper they state the probabilities used in the rejection criteria should have
    # top-p already applied (if using top-p)
    return g


def main(
    prompt: str = "Alan Turing theorized that computers would one day become",
    n_tokens_to_generate: int = 20,
    draft_model_size: str = "124M",
    target_model_size: str = "1558M",
    models_dir: str = "models",
    K: int = 4,
    seed: int = 123,
):
    # seed numpy rng
    np.random.seed(seed)

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, draft_hparams, draft_params = load_encoder_hparams_and_params(
        draft_model_size, models_dir
    )
    _, target_hparams, target_params = load_encoder_hparams_and_params(
        target_model_size, models_dir
    )
    draft_model = create_model_fn(draft_params, draft_hparams)
    target_model = create_model_fn(target_params, target_hparams)

    # encode inputs
    input_ids = encoder.encode(prompt)

    def run_sampling_fn(decode_fn, input_ids, **kwargs):
        start = time.perf_counter()
        output_ids = decode_fn(x=input_ids, **kwargs)
        text = encoder.decode(output_ids)
        elapsed_time = time.perf_counter() - start
        return text, elapsed_time

    # autoregressive
    autoregressive_text, autoregressive_time = run_sampling_fn(
        autoregressive_sampling,
        input_ids,
        model=target_model,
        N=n_tokens_to_generate,
    )
    print("Autoregressive Decode")
    print("--------------")
    print(f"Time = {autoregressive_time:.2f}s")
    print(f"Text = {autoregressive_text}")
    print()

    # speculative
    speculative_text, speculative_time = run_sampling_fn(
        speculative_sampling,
        input_ids,
        target_model=target_model,
        draft_model=draft_model,
        N=n_tokens_to_generate,
        K=K,
    )
    print("Speculative Decode")
    print("------------------")
    print(f"Time = {speculative_time:.2f}s")
    print(f"Text = {speculative_text}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
