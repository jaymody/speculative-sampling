# Speculative Sampling
A simple implementation of [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318.pdf) in NumPy for GPT-2. See [`main.py`](https://github.com/jaymody/speculative-sampling/blob/main/main.py). I also wrote a [blog post](https://jaykmody.com/blog/speculative-sampling/) for this implementation.

**Install Dependencies**:
```bash
pip install -r picoGPT/requirements.txt
```
Tested on `Python 3.9.10`.

**Usage**:
```python
python main.py \
    --prompt "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --draft_model_size "124M" \
    --target_model_size "1558M" \
    --K 4 \
    --temperature 0 # 0 for greedy sampling
```

Which outputs:
```text
Autoregressive Decode
---------------------
Time = 60.64s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T

Speculative Decode
------------------
Time = 27.15s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T
```
