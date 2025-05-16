# ThinkGeo: Evaluating Tool-Augmented Agents for Remote Sensing Tasks

## Introduction

ThinkGeo is a specialized benchmark designed to evaluate how language model agents handle complex remote sensing tasks through structured tool use and step-by-step reasoning. It features human-curated queries grounded in satellite and aerial imagery across diverse real-world domains such as disaster response, urban planning, and environmental monitoring. Using a ReAct-style interaction loop, ThinkGeo tests both open and closed-source LLMs on over 400 multi-step agentic tasks. The benchmark measures not only final answer correctness but also the accuracy and consistency of tool usage throughout the process. By focusing on spatially grounded, domain-specific challenges, ThinkGeo fills a critical gap left by general-purpose evaluation frameworks.

## 🚀 Evaluating ThinkGeo

### Prepare ThinkGeo Dataset
1. Clone this repo.
```shell
git clone https://github.com/anonneurips8/ThinkGeo.git
cd ThinkGeo
```

2. Download the dataset from [Hugging face ThinkGeo](https://huggingface.co/datasets/anonneurips8/ThinkGeo).
```shell
mkdir ./opencompass/data
```
Put it under the folder ```./opencompass/data/```. The structure of files should be:
```
ThinkGeo/
├── agentlego
├── opencompass
│   ├── data
│   │   ├── ThinkGeo_dataset
│   ├── ...
├── ...
```

### Prepare Your Model
1. Download the model weights.
```shell
pip install -U huggingface_hub
# huggingface-cli download --resume-download hugging/face/repo/name --local-dir your/local/path --local-dir-use-symlinks False
huggingface-cli download --resume-download Qwen/Qwen1.5-7B-Chat --local-dir ~/models/qwen1.5-7b-chat --local-dir-use-symlinks False
```

2. Install [LMDeploy](https://github.com/InternLM/lmdeploy).
```shell
conda create -n lmdeploy python=3.10
conda activate lmdeploy
```
For CUDA 12:
```shell
pip install lmdeploy
```

3. Launch a model service.
```shell
# lmdeploy serve api_server path/to/your/model --server-port [port_number] --model-name [your_model_name]
lmdeploy serve api_server ~/models/qwen1.5-7b-chat --server-port 12580 --model-name qwen1.5-7b-chat
```
### Deploy Tools
1. Install [AgentLego](https://github.com/InternLM/agentlego).
```shell
conda create -n agentlego python=3.11.9
conda activate agentlego
cd agentlego
pip install -r requirements_all.txt
pip install agentlego
pip install -e .
mim install mmengine
mim install mmcv==2.1.0
```
Open ```~/anaconda3/envs/agentlego/lib/python3.11/site-packages/transformers/modeling_utils.py```, then set ```_supports_sdpa = False``` to ```_supports_sdpa = True``` in line 1279.

2. Deploy tools for ThinkGeo benchmark.

To use the GoogleSearch, you should first get the Serper API key from https://serper.dev . Then export this keys as environment variables.

```shell
export SERPER_API_KEY='your_serper_key_for_google_search_tool'
```

Start the tool server.

```shell
agentlego-server start --port 16181 --extra ./benchmark.py  `cat benchmark_toollist.txt` --host 0.0.0.0
```
### Start Evaluation
1. Install [OpenCompass](https://github.com/open-compass/opencompass).
```shell
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
cd agentlego
pip install -e .
cd ../opencompass
pip install -e .
```
huggingface_hub==0.25.2 (<0.26.0)
transformers==4.40.1
2. Modify the config file at ```configs/eval_ThinkGeo_bench.py``` as below.

The ip and port number of **openai_api_base** is the ip of your model service and the port number you specified when using lmdeploy.

The ip and port number of **tool_server** is the ip of your tool service and the port number you specified when using agentlego.

```python
models = [
  dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        tool_server='http://10.140.0.138:16181',
        tool_meta='data/ThinkGeo_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```

If you infer and evaluate in **step-by-step** mode, you should comment out **tool_server** and enable **tool_meta** in ```configs/eval_ThinkGeo_bench.py```, and set infer mode and eval mode to **every_with_gt** in ```configs/datasets/ThinkGeo_bench.py```:
```python
models = [
  dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        # tool_server='http://10.140.0.138:16181',
        tool_meta='data/ThinkGeo_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```
```python
ThinkGeo_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every_with_gt'),
)
ThinkGeo_bench_eval_cfg = dict(evaluator=dict(type=ThinkGeoBenchEvaluator, mode='every_with_gt'))
```

If you infer and evaluate in **end-to-end** mode, you should comment out **tool_meta** and enable **tool_server** in ```configs/eval_ThinkGeo_bench.py```, and set infer mode and eval mode to **every** in ```configs/datasets/ThinkGeo_bench.py```:
```python
models = [
  dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        tool_server='http://10.140.0.138:16181',
        # tool_meta='data/ThinkGeo_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```
```python
ThinkGeo_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every'),
)
ThinkGeo_bench_eval_cfg = dict(evaluator=dict(type=ThinkGeoBenchEvaluator, mode='every'))
```

3. Infer and evaluate with OpenCompass.
```shell
# infer only
python run.py configs/eval_ThinkGeo_bench.py --max-num-workers 32 --debug --mode infer
```
```shell
# evaluate only
# srun -p llmit -q auto python run.py configs/eval_ThinkGeo_bench.py --max-num-workers 32 --debug --reuse [time_stamp_of_prediction_file] --mode eval
srun -p llmit -q auto python run.py configs/eval_ThinkGeo_bench.py --max-num-workers 32 --debug --reuse 20250616_112233 --mode eval
```
```shell
# infer and evaluate
python run.py configs/eval_ThinkGeo_bench.py -p llmit -q auto --max-num-workers 32 --debug
```
