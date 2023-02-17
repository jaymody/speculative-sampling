# Speculative Sampling
Simple and minimal implementation of [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318.pdf) in NumPy for GPT-2. See [`main.py`](https://github.com/jaymody/speculative-sampling/blob/main/main.py). I also wrote a [blog post](https://jaykmody.com/blog/speculative-sampling) for this implementation.

GPT-2 code adapted from [picoGPT](https://github.com/jaymody/picoGPT). For the decoding algorithm themselves, I try and match the notation used in the paper as much as possible.

This implementation is for demonstrative purposes (i.e. extremely minimal, for example KV caching/batching/top-p are not implemented). As such, I wouldn't pay too much attention to the speedup times. I also haven't verified that there is no performance degradation outside of some qualitative assessment. I use GPT-2 1558M as the target model and GPT-2 124M as the draft model.

**Install Dependencies**:
```bash
pip install -r picoGPT/requirements.txt
```
If you're using an M1 Macbook, you'll need to replace tensorflow with `tensorflow-macos`.

Tested on `Python 3.9.10`.

**Usage**:
```python
python speculative_sampling.py \
    --prompt "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --draft_model_size "124M" \
    --target_model_size "1558M" \
    --K 4
```

Which outputs:
```text
Autoregressive Decode
---------------------
Time = 71.64s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T

Speculative Decode
------------------
Time = 30.11s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think for themselves. But it's not just computers that are capable of thinking for themselves.

In fact, the brain is a computer, and it's capable
```
