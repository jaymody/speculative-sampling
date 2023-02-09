# Speculative Sampling
Simple and minimal implementation of [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318.pdf) in NumPy for GPT-2. See `main.py`.

GPT-2 code adapted from [picoGPT](https://github.com/jaymody/picoGPT). For the decoding algorithm themselves, I try and match the notation used in the paper as much as possible.

This implementation is for demonstrative purposes (i.e. extremely minimal, for example KV caching/batching is not implemented, and of course, there is no GPU support). As such, I wouldn't pay too much attention to the speedup times.

**Install Dependencies**:
```bash
pip install -r requirements.txt
```
If you're using an M1 Macbook, you'll need to replace tensorflow with `tensorflow-macos`.

Tested on `Python 3.9.10`.

**Usage**:
```python
python main.py \
    --prompt "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40
```

This will take a while (about 5-10 minutes on an M1 MacBook). It outputs:
```
Autoregressive Decode
--------------
Time = 204.80s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T

Speculative Decode
------------------
Time = 90.88s
Text = Alan Turing theorized that computers would one day become the most powerful machines on the planet. But the technology was not ready for prime time.

"The computer is not ready for prime time," he said. "It's not even ready for
```
