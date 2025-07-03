# Skywork-Reward-V2

<div align="center">
  <img src="assets/skywork_logo.png" width="60%" alt="Skywork-Reward-V2"/>
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2507.01352" target="_blank">
    <img alt="Paper" src="https://img.shields.io/badge/📖_Paper-Skywork--Reward--V2-4D5EFF?style=flat-square&labelColor=202124"/>
  </a>
  <a href="https://huggingface.co/collections/Skywork/skywork-reward-v2-685cc86ce5d9c9e4be500c84" target="_blank">
    <img alt="Models" src="https://img.shields.io/badge/🤗_Hugging_Face-Model_Collection-4D5EFF?style=flat-square&labelColor=202124"/>
  </a>
  <a href="https://github.com/SkyworkAI/Skywork-Reward-V2" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/🧑‍💻_GitHub-Skywork--Reward--V2-4D5EFF?style=flat-square&labelColor=202124"/>
  </a>
</div>

## 🔥 Highlights

**Skywork-Reward-V2** is a series of eight reward models designed for versatility across a wide range of tasks, trained on a mixture of 26 million carefully curated preference pairs. While the Skywork-Reward-V2 series remains based on the Bradley-Terry model, we push the boundaries of training data scale and quality to achieve superior performance. Compared to the first generation of Skywork-Reward, the Skywork-Reward-V2 series offers the following major improvements:

- **Trained on a significantly larger and higher-quality preference data mixture**, consisting of **26 million preference pairs** curated via a large-scale human-LLM synergistic pipeline.
- **State-of-the-art performance on seven major reward model benchmarks**, including RewardBench v1, RewardBench v2, PPE Preference, PPE Correctness, RMB, RM-Bench, and JudgeBench.
- **Available in eight models across multiple sizes**, with the smallest 0.6B variant, *Skywork-Reward-V2-Qwen3-0.6B*, nearly matching the average performance of our previous best model, Skywork-Reward-Gemma-2-27B-v0.2. The largest 8B version, *Skywork-Reward-V2-Llama-3.1-8B*, surpasses all existing reward models across all benchmarks on average. Our top experimental model, *Skywork-Reward-V2-Llama-3.1-8B-40M*, **outperforms all existing reward models on every benchmark**.

<div align="center">

| Model                              | Base Model                                                                                  |                                         Link                                         |
|:-----------------------------------|:--------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------:|
| Skywork-Reward-V2-Llama-3.1-8B     | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |   [🤗 Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)   |
| Skywork-Reward-V2-Llama-3.1-8B-40M | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [🤗 Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M) |
| Skywork-Reward-V2-Llama-3.2-1B     | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |   [🤗 Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.2-1B)   |
| Skywork-Reward-V2-Llama-3.2-3B     | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |   [🤗 Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.2-3B)   |
| Skywork-Reward-V2-Qwen3-0.6B       | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                                   |    [🤗 Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)    |
| Skywork-Reward-V2-Qwen3-1.7B       | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)                                   |    [🤗 Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-1.7B)    |
| Skywork-Reward-V2-Qwen3-4B         | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)                                       |     [🤗 Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B)     |
| Skywork-Reward-V2-Qwen3-8B         | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)                                       |     [🤗 Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)     |

</div>

For the complete collection of models, please refer to the [Skywork-Reward-V2](https://huggingface.co/collections/Skywork/skywork-reward-v2-685cc86ce5d9c9e4be500c84) collection.

## 📊 Evaluation

In the following table, we categorize the models into two types: Bradley-Terry (BT) reward models and Generative reward models. The Skywork-Reward-V2 series outperforms models in both categories with much smaller model sizes.

|     Category      | Model                                  | RewardBench v1 | RewardBench v2 | PPE Preference | PPE Correctness |   RMB    | RM-Bench | JudgeBench |   Avg.   |
|:-----------------:|:---------------------------------------|:--------------:|:--------------:|:--------------:|:---------------:|:--------:|:--------:|:----------:|:--------:|
| **Bradley-Terry** | Llama-3-OffsetBias-RM-8B               |      89.0      |      64.8      |      59.2      |      64.1       |   57.8   |   71.3   |    63.5    |   67.1   |
|                   | ArmoRM-Llama3-8B-v0.1                  |      90.4      |      66.5      |      60.6      |      60.6       |   64.6   |   69.3   |    59.7    |   67.4   |
|                   | Internlm2-20b-reward                   |      90.2      |      56.3      |      61.0      |      63.0       |   62.9   |   68.3   |    64.3    |   66.6   |
|                   | Skywork-Reward-Llama-3.1-8B-v0.2       |      93.1      |      71.8      |      62.2      |      62.5       |   66.6   |   72.1   |    62.9    |   70.2   |
|                   | LDL-Reward-Gemma-2-27B-v0.1            |      95.0      |      72.5      |      62.4      |      63.9       |   67.9   |   71.1   |    64.2    |   71.0   |
|                   | Skywork-Reward-Gemma-2-27B-v0.2        |      94.3      |      75.3      |      63.6      |      61.9       |   69.4   |   70.0   |    66.5    |   71.6   |
|                   | INF-ORM-Llama3.1-70B                   |      95.1      |      76.5      |      64.2      |      64.4       |   70.5   |   73.8   |    70.2    |   73.5   |
|  **Generative**   | GPT-4o                                 |      86.7      |      64.9      |      67.7      |        -        |   73.8   |    -     |    59.8    |    -     |
|                   | Claude-3.5-Sonnet                      |      84.2      |      64.7      |      67.3      |        -        |   70.6   |    -     |    64.8    |    -     |
|                   | DeepSeek-GRM-27B                       |      88.5      |       -        |      65.3      |      60.4       |   69.0   |    -     |     -      |    -     |
|                   | DeepSeek-GRM-27B (w/ MetaRM)           |      90.4      |       -        |      67.2      |      63.2       |   70.3   |    -     |     -      |    -     |
|                   | RM-R1-Qwen-Instruct-32B                |      92.9      |       -        |       -        |        -        |   73.0   |   79.1   |     -      |    -     |
|                   | RM-R1-DeepSeek-Distill-Qwen-32B        |      90.9      |       -        |       -        |        -        |   69.8   |   83.9   |     -      |    -     |
|                   | EvalPlanner (Llama-3.1-70B)            |      93.9      |       -        |       -        |        -        |    -     |   80.0   |    50.9    |    -     |
|                   | EvalPlanner (Llama-3.3-70B)            |      93.8      |       -        |       -        |        -        |    -     |   82.1   |    56.6    |    -     |
|                   | J1-Llama-8B                            |      85.7      |       -        |      60.3      |      59.2       |    -     |   73.4   |    42.0    |    -     |
|                   | J1-Llama-8B (Maj@32)                   |       -        |       -        |      60.6      |      61.9       |    -     |    -     |     -      |    -     |
|                   | J1-Llama-70B                           |      93.3      |       -        |      66.3      |      72.9       |    -     |   82.7   |    60.0    |    -     |
|                   | J1-Llama-70B (Maj@32)                  |       -        |       -        |      67.0      |      73.7       |    -     |    -     |     -      |    -     |
| **Bradley-Terry** | **Skywork-Reward-V2-Qwen3-0.6B**       |      85.2      |      61.3      |      65.3      |      68.3       |   74.5   |   74.4   |    67.6    |   70.9   |
|                   | **Skywork-Reward-V2-Qwen3-1.7B**       |      90.3      |      68.3      |      67.6      |      70.5       |   78.1   |   78.7   |    72.9    |   75.2   |
|                   | **Skywork-Reward-V2-Qwen3-4B**         |      93.4      |      75.5      |      69.5      |      74.7       |   80.6   |   81.6   |    69.3    |   77.8   |
|                   | **Skywork-Reward-V2-Qwen3-8B**         |      93.7      |      78.2      |      70.6      |      75.1       |   81.2   |   82.6   |    73.4    |   79.3   |
|                   | **Skywork-Reward-V2-Llama-3.2-1B**     |      89.9      |      64.3      |      66.6      |      67.4       |   76.7   |   76.4   |    65.0    |   72.3   |
|                   | **Skywork-Reward-V2-Llama-3.2-3B**     |      93.0      |      74.7      |      69.1      |      72.1       |   80.5   |   81.1   |    69.2    |   77.1   |
|                   | **Skywork-Reward-V2-Llama-3.1-8B**     |      96.4      |      84.1      |      77.3      |      83.4       |   86.4   |   92.8   |    80.0    |   85.8   |
|                   | **Skywork-Reward-V2-Llama-3.1-8B-40M** |    **97.8**    |    **86.5**    |    **79.8**    |    **87.2**     | **89.3** | **96.0** |  **83.4**  | **88.6** |

## 💡 Recommended Usage

We make the following recommendations for using the Skywork-Reward-V2 model series:

1. For most use cases, we recommend Skywork-Reward-V2-Llama-3.1-8B and consider smaller variants for low-resource settings.
2. All models are trained on preference data with a maximum length of 16,384 tokens. It is recommended to perform inference within this limit.
3. Do not include system prompts when using chat templates.

Special note on Skywork-Reward-V2-Llama-3.1-8B-40M:

> [!NOTE]
> Although Skywork-Reward-V2-Llama-3.1-8B-40M outperforms the original Skywork-Reward-V2-Llama-3.1-8B, we consider it an experimental variant. This model is trained on the complete set of 40 million preference pairs, with about one third of the chosen-rejected pairs flipped. We recommend using this model solely for research or non-production purposes.

## 📦 Model Usage

### 📝 Simple Example in `transformers`

The example below shows how to perform inference in Hugging Face Transformers to get the reward score for conversations. For better data parallelization and throughput, we recommend using it along with [Accelerate](https://github.com/huggingface/accelerate) if multiple GPUs are available.

```python
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]

# Format and tokenize the conversations
conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
conv2_formatted = tokenizer.apply_chat_template(conv2, tokenize=False)
# These two lines remove the potential duplicate bos token
if tokenizer.bos_token is not None and conv1_formatted.startswith(tokenizer.bos_token):
    conv1_formatted = conv1_formatted[len(tokenizer.bos_token):]
if tokenizer.bos_token is not None and conv2_formatted.startswith(tokenizer.bos_token):
    conv2_formatted = conv2_formatted[len(tokenizer.bos_token):]
conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt").to(device)
conv2_tokenized = tokenizer(conv2_formatted, return_tensors="pt").to(device)

# Get the reward scores
with torch.no_grad():
    score1 = rm(**conv1_tokenized).logits[0][0].item()
    score2 = rm(**conv2_tokenized).logits[0][0].item()
print(f"Score for response 1: {score1}")
print(f"Score for response 2: {score2}")

# Expected output:
# Score for response 1: 23.0
# Score for response 2: 3.59375
```

### ⚡ Distributed Inference via SGLang

For the optimal throughput under a large number (e.g., millions) of conversations, we recommend the following distributed method via SGLang.

Install the latest version of [SGLang](https://docs.sglang.ai/index.html):

```bash
pip install "sglang[all]>=0.4.7.post1"
```

Launch model servers (assuming `NUM_GPUS` GPUs are available):

```bash
NUM_GPUS=8
for (( i=0; i<NUM_GPUS; i++ )); do
    echo "Starting server on port $((8000+i)) with GPU: $i"
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path Skywork/Skywork-Reward-V2-Llama-3.1-8B \
        --mem-fraction-static 0.9 \
        --tp 1 \
        --host 127.0.0.1 \
        --port $((8000+i)) \
        --context-length 16384 \
        --is-embedding \
        &
done
```

After the servers are ready, we can get rewards from the servers. You should get reward values similar to those in the `transformers` example above.

```python
import requests
from transformers import AutoTokenizer


model_name_or_path = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
base_urls = [f"http://127.0.0.1:{8000 + i}/classify" for i in range(8)]
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def process_convs(convs, base_url, tokenizer, model_name_or_path):
    payload = {"model": model_name_or_path}
    convs_formatted = []
    for conv in convs:
        conv = tokenizer.apply_chat_template(conv, tokenize=False)
        if tokenizer.bos_token is not None and conv.startswith(tokenizer.bos_token):
            conv = conv[len(tokenizer.bos_token) :]
        convs_formatted.append(conv)

    payload.update({"text": convs_formatted})
    rewards = []
    try:
        responses = requests.post(base_url, json=payload).json()
        for response in responses:
            rewards.append(response["embedding"][0])
        assert len(rewards) == len(
            convs
        ), f"Expected {len(convs)} rewards, got {len(rewards)}"
        return rewards
    except Exception as e:
        print(f"Error: {e}")
        return [None] * len(convs)
    

prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]

rewards = process_convs([conv1, conv2], base_urls[0], tokenizer, model_name_or_path)
print(f"Score for response 1: {rewards[0]}")
print(f"Score for response 2: {rewards[1]}")

# Expected output:
# Score for response 1: 23.125
# Score for response 2: 3.578125
```

## 📃 License

Reward models in the Skywork-Reward-V2 series derived from Qwen3 support commercial use and permit modifications and the creation of derivative works, provided that all conditions of the Apache 2.0 License are met and proper attribution is given. Please note that:

- Skywork-Reward-V2-Qwen3-0.6B, Skywork-Reward-V2-Qwen3-1.7B, Skywork-Reward-V2-Qwen3-4B, and Skywork-Reward-V2-Qwen3-8B are derived from the Qwen3 model series of corresponding sizes, which are originally licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
- Skywork-Reward-V2-Llama-3.1-8B and Skywork-Reward-V2-Llama-3.1-8B-40M are both derived from Llama-3.1-8B-Instruct and follow the [Llama 3.1 community license](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE).
- Skywork-Reward-V2-Llama-3.2-1B and Skywork-Reward-V2-Llama-3.2-3B are derived from Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct, respectively, and follow the [Llama 3.2 community license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt).

## 📧 Contact

If you have any questions, please feel free to reach us at `yuhao.liuu at kunlun-inc dot com` and `liang.zeng at kunlun-inc dot com`.

## 📚 Citation

If you find our work useful, please cite it as follows.

```bibtex
@article{liu2025skywork,
  title={Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy},
  author = {Liu, Chris Yuhao and Zeng, Liang and Xiao, Yuzhen and He, Jujie and Liu, Jiacai and Wang, Chaojie and Yan, Rui and Shen, Wei and Zhang, Fuxiang and Xu, Jiacheng and Liu, Yang and Zhou, Yahui},
  journal={arXiv preprint arXiv:2507.01352},
  year={2025}
}
```
