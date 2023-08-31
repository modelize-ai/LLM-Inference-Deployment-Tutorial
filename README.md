<h1 align="center">大语言模型推理部署教程</h1>
<p align="center">面向大语言模型开发工程师的，囊括推理引擎设计、服务部署、性能评测等方面的进阶教程，并开源了一套经过优化的单卡推理引擎代码供研究与学习</p>

<p align="center">教程文档正在持续快速地更新中，watch 本项目以第一时间获取最新内容！</p>

## 目录

- [本项目文件结构](#本项目文件结构)
- [推理引擎代码](#推理引擎代码)
  * [介绍](#介绍)
  * [环境配置](#环境配置)
  * [服务部署](#服务部署)


## 本项目文件结构

本项目主要包含两个目录：`code` 和 `docs`。
- `code` 目录：存放着我们以研究和学习为主要目的的推理引擎代码，开发者们可以在该目录下通过简单几步来快速配置环境和部署服务，以对比和体验不同策略在不同流量特征下的差异。
- `docs` 目录：存放着所有章节的教程文档，通过阅读相关内容，开发者们可以了解到在商业化的生产环境中部署大语言模型进行推理的多方面知识。

## 推理引擎代码

### 介绍

我们在本项目中开源了一套经过优化的大语言模型单卡推理引擎，采用 C/S 设计模式，向开发者们展示大语言模型推理引擎的内部运作机理，方便大家亲自上手部署以体验不同策略在不同流量特征下的性能差异。

#### 优点

- 加载 `int4` GPTQ 模型
- （客户端）负载均衡
- 多策略的服务端：**静态批处理** (Static Batching, SB) 策略和**持续批处理** (Continuous Batching, CB) 策略
- （支持持续批处理的服务端）基于 xformers 和 PagedAttention 加速推理
- （支持持续批处理的服务端）异构解码策略
- GPU-CPU 内存交换

#### 缺点

- 多卡推理
- 流式传输
- （支持持续批处理的服务端）只支持 llama v1 和 llama v2（除 70B）模型
- （支持持续批处理的服务端）未对注意力层之外的其他网络模块和计算操作进行推理性能的优化
- （支持持续批处理的服务端）仅支持 `safetensors` 格式的权重文件
- （支持持续批处理的服务端）不支持拓展最大可处理的上下文长度

#### 其他开源大模型推理框架

本项目提供的推理引擎代码旨在向开发者们展示大语言模型推理引擎的内部运作机理，因此未对与这一目的关系较弱或无关的方面做进一步优化。

通过使用我们提供的这套推理引擎代码，开发者们可以在单卡上轻松部署 20B 及以下的模型，但在面对更大参数量的模型时则捉襟见肘。为此，我们在这里列出开源社区上现已存在的功能较为完善的大语言模型推理框架并作简单介绍：

- [TGI](https://github.com/huggingface/text-generation-inference): Hugging Face 在内部生产环境中使用的大语言模型推理框架，注意在 0.9.4 版本之后应用于商业目的需获得官方许可；
- [vLLM](https://github.com/vllm-project/vllm): 高性能、易用的大语言模型推理框架，提出并实现了 PagedAttention；
- [lmdeploy](https://github.com/InternLM/lmdeploy): 书生浦语团队研发的大语言模型推理部署框架；
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine): 英伟达新一代 Transformers 架构模型推理框架，支持 `fp8` 格式；
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII): 基于 DeepSpeed 的低延迟和高吞吐推理引擎，不仅仅只支持大语言模型。

此外，还有 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), [FlexGen](https://github.com/FMInference/FlexGen), [EnergonAI](https://github.com/hpcaitech/EnergonAI) 等针对大语言模型推理性能优化而设计的框架，但更新频率较低，故不在此作进一步介绍。

我们十分建议开发者们在学习完本项目后，亲自去阅读以上开源项目的代码，以进一步理解大模型推理引擎设计思路和优化技术。

> 本项目提供的推理引擎代码在权重加载、模型网络代码设计和文本生成策略实现上借鉴了 TGI 项目；在内存管理上借鉴了 vLLM 项目。

### 环境配置

#### 客户端

在 `code` 目录下，执行 `pip install -r client_requirements.txt` 来安装客户端所需的第三方代码库。

#### 服务端

首先，确保虚拟环境中已经安装了支持 CUDA 的 2.0.0 及以上版本的 PyTorch，若没有，可以在 [这里](https://pytorch.org/get-started/locally/) 选择并安装与你的软硬件信息相符的预编译安装包，也可以根据 [这里](https://github.com/pytorch/pytorch#from-source) 的指导从源码编译和安装。

其次，安装 vLLM，此举的目的是为了方便我们在代码中使用 paged-attention 算子和与内存管理相关的算子。
- 快速安装：`pip install vllm`；
- 从源码编译安装：遵循 [这里](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source) 的指导。

接着，根据 [这里](https://github.com/PanQiWei/AutoGPTQ#installation) 的指导安装 `auto-gptq` 代码库，我们提供的推理引擎支持加载和使用 GPTQ 模型。

最后，在 `code` 目录下，执行 `pip install -r server_requirements.txt` 安装服务端所需的其他第三方代码库。

### 服务部署
正如你所见，`code` 目录下包含了四个模块：
- `protocol`：定义了客户端和服务端之间的通信规范；
- `client`：客户端相关代码；
- `server`：服务端相关代码；
- `utils`：构建 HTTP 服务代码时的其他工具代码。

你可以组合这些模块并将它们封装成单独的 HTTP 服务，如：
- `client_app.py`：使用 `protocol`, `client` 和 `utils` 模块构建的客户端 HTTP 服务；
- `continuous_batching_server_app.py` 和 `static_batching_server_app.py`：使用 `protocol`, `server` 和 `utils` 模块构建的服务端 HTTP 服务。

也可以将这些模块直接拷贝到你当前工作的项目中进行集成应用。

我们在 `code` 目录中提供了三个以 "_app.py" 为后缀的 HTTP 服务脚本，你可以直接执行这些脚本来启动服务，如：
```shell
CUDA_VISIBLE_DEVICES=0 python client_app.py --port 8000  # 查询更多命令行参数可使用 python client_app.py --help
```

此时服务进程将始终运行在前台，你可以直接使用 `ctrl`+`c` 来关闭进程。

此外，我们还提供了一个统一的服务部署脚本 `start_app.py`，通过运行这个脚本，你可以使用 `gunicorn` 来管理和在后台运行服务进程。

> 注意：由于我们提供的推理引擎目前只支持单卡推理，因此在运行服务脚本时，强烈建议同时设置 CUDA_VISIBLE_DEVICES 环境变量以使用指定的单张显卡，避免预期外的行为发生。

部署每个 HTTP 服务都需要提供一个相应的配置文件，以下为各服务配置文件模板（使用时注意删去注释文本）：

<details>
<summary>client_config.json</summary>

```json
{
  "continuous_batching_server_urls": ["http://127.0.0.1:8001"],  # 支持持续批处理的服务端 HTTP 服务地址，请求会在此间负载均衡地分发
  "static_batching_server_urls": ["http://127.0.0.1:8002"],  # 支持静态批处理的服务端 HTTP 服务地址，请求会在此间负载均衡地分发
  "openai_jumper_configs": [
    {
      "api_key": "YOUR_OPENAI_KEY",
      "org_id": null
    }
  ],  # openai 账号列表，请求会在此间负载均衡地分发
  "heart_beat_interval_seconds": 600  # 对服务端的 HTTP 服务心跳检测间隔（以秒为单位）
}
```
</details>

<details>
<summary>cb_server_config.json</summary>

```json
{
  "model_loading_config": {
    "model_type": "llama",  # 模型架构类型，目前只支持 llama
    "model_name_or_path": "PATH_TO_MODEL_DIR",  # 存放模型权重文件的目录路径，只支持 safetensors 格式
    "torch_dtype": "float16",  # （非 GPTQ 模型时）模型权重和运算时使用的数值类型，可选项为 float16 和 bfloat16
    "tokenizer_name_or_path": null,  # 存放分词器模型文件的目录路径，如果为空则使用存放模型权重文件的目录路径
    "use_fast_tokenizer": false,  # 若为 true 则加载分词器时设置 use_fast=True
    "trust_remote_code": false,  # 是否使用非 Hugging Face 官方提供的模型或分词器代码
    "quantize_method": null,  # 量化方法，可选值为 gptq
    "model_max_length": 2048,  # 模型能处理的最大上下文长度
    "gptq_model_base_name": null,  # GPTQ 模型权重文件名称（不包含文件拓展名），若为空则使用默认的命名格式查找文件
    "gptq_config_base_name": null  # GPTQ 配置文件名称（不包含文件拓展名），若为空则使用默认的命名格式查找文件
  },
  "batcher_config": {
    "batch_max_tokens": 56000,  # 一个批次中同时处理的最大 tokens 数量，这里的值为 llama-7b fp16 模型在 A100-40G 上的一个合理值
    "batch_max_beams": 32  # 一个批次中同时处理的最大 beam（文本生成阶段的预测分支） 数量，这里的值为 llama-7b fp16 模型在 A100-40G 上的一个合理值
  },
  "cache_config": {
    "num_blocks": 2500,  # GPU 内存块数量，这里的值为 llama-7b fp16 模型在 A100-40G 上的一个合理值
    "num_blocks_cpu": 1024,  # CPU 内存块数量
    "block_size": 16,  # 一个内存块的大小
    "watermark": 0.01,  # 预留的 GPU 内存块比例，这是为了防止过分分配 GPU 内存块给 prompt 的 和从 CPU 内存换入的 KV Cache 而导致文本生成时 GPU 内存资源紧张
  }
}
```
</details>

<details>
<summary>sb_server_config.json</summary>

```json
{
  "batcher_config": {
    "package_max_workload": "16",  # 一个任务包的最大工作负载，单位为 beam
    "packaging_interval_seconds": 2  # 打包的间隔时间，TPS/QPS 越小间隔时间可以越长
  },
  "worker_config": {
    "model_name_or_path": "PATH_TO_MODEL_DIR",  # 存放模型权重文件的目录路径
    "tokenizer_name_or_path": null,  # 存放模型权重文件的目录路径
    "revision": "main",  # 使用的模型仓库分支，仅在目录路径为 Hugging Face Hub 模型名或 github 仓库目录时生效
    "low_cpu_mem_usage": true,  # 是否直接将模型权重加载到 GPU 
    "torch_dtype": "float16",  # （非 GPTQ 模型时）模型权重和运算时使用的数值类型，可选项为 float16 和 bfloat16
    "use_fast_tokenizer": false,  # 若为 true 则加载分词器时设置 use_fast=True
    "trust_remote_code": false,  # 是否使用非 Hugging Face 官方提供的模型或分词器代码
    "use_safetensors": false,  # 是否加载 `safetensors` 格式的权重文件
    "batch_size": -1,  # TextGenerationPipeline 执行时所使用的 batch_size，-1 表示同时处理所有的输入
    "is_gptq_quantized": false  # 是否使用的是 GPTQ 模型
  }
}
```
</details>
