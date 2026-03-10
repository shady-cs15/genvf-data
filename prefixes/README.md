## prefix collection v0

link: https://huggingface.co/datasets/haoranli-ml/genvf-prefixes-filtered

1. stats of this data
    - num of problems: 3000
    - num of prefixes: 6000
    - 2 prefixes for each question, coming from [qwen3-235b, gpt5-mini/gemini-3-flash/gemini-3-pro]

2. descriptions: for each problem, one prefix always comes from qwen3-235b, another prefix is randomly selected from ['gpt5-mini', 'gemini-3-flash', 'gemini-3-pro']. The reason for this operation is that proprietary models don't expose their raw cots. Instead, they only expose their summarized reasoning. Therefore, the cots from these proprietary models are much shorter than that of qwen models. To balance prefix length, we use this strategy.


## prefix collection v1: haoranli-ml/genvf-prefixes-v1

link: https://huggingface.co/datasets/haoranli-ml/genvf-prefixes-v1

1. stats of this data
    - num of problems: 3000
    - num of prefixes: 12000
    - 4 prefixes for each question, coming from [qwen3-235b, gpt5-mini/gemini-3-flash/gemini-3-pro, qwen3.5-27b, qwen3.5-35b-a3b]
