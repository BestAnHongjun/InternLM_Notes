# 【Lecture 6】OpenCompass大模型评测

## 一、关于评测的三个问题

### 1.为什么要评测？
* 模型选型
* 模型能力提升
* 真是应用场景效果评测

### 2.我们需要测什么？
* 知识、推理、语言
* 长文本、智能体、多轮对话
* 情感、认知、价值观

### 3.怎么样测试大语言模型？
* 自动化客观评测
* 人机交互评测
* 基于大模型的大模型评测

## 二、实战部分

### 1.安装

#### (1)面向GPU的环境安装

```sh

conda create --name opencompass --clone=/root/share/conda_envs/internlm-base
source activate opencompass
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

#### (2)数据准备

```sh
# 解压评测数据集到 data/ 处
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip

# 将会在opencompass下看到data文件夹
```

#### (3)查看支持的数据集和模型

```sh
# 列出所有跟 internlm 及 ceval 相关的配置
python tools/list_configs.py internlm ceval
```

将会看到

```sh
+--------------------------+--------------------------------------------------------+
| Model                    | Config Path                                            |
|--------------------------+--------------------------------------------------------|
| hf_internlm_20b          | configs/models/hf_internlm/hf_internlm_20b.py          |
| hf_internlm_7b           | configs/models/hf_internlm/hf_internlm_7b.py           |
| hf_internlm_chat_20b     | configs/models/hf_internlm/hf_internlm_chat_20b.py     |
| hf_internlm_chat_7b      | configs/models/hf_internlm/hf_internlm_chat_7b.py      |
| hf_internlm_chat_7b_8k   | configs/models/hf_internlm/hf_internlm_chat_7b_8k.py   |
| hf_internlm_chat_7b_v1_1 | configs/models/hf_internlm/hf_internlm_chat_7b_v1_1.py |
| internlm_7b              | configs/models/internlm/internlm_7b.py                 |
| ms_internlm_chat_7b_8k   | configs/models/ms_internlm/ms_internlm_chat_7b_8k.py   |
+--------------------------+--------------------------------------------------------+
+----------------------------+------------------------------------------------------+
| Dataset                    | Config Path                                          |
|----------------------------+------------------------------------------------------|
| ceval_clean_ppl            | configs/datasets/ceval/ceval_clean_ppl.py            |
| ceval_gen                  | configs/datasets/ceval/ceval_gen.py                  |
| ceval_gen_2daf24           | configs/datasets/ceval/ceval_gen_2daf24.py           |
| ceval_gen_5f30c7           | configs/datasets/ceval/ceval_gen_5f30c7.py           |
| ceval_ppl                  | configs/datasets/ceval/ceval_ppl.py                  |
| ceval_ppl_578f8d           | configs/datasets/ceval/ceval_ppl_578f8d.py           |
| ceval_ppl_93e5ce           | configs/datasets/ceval/ceval_ppl_93e5ce.py           |
| ceval_zero_shot_gen_bd40ef | configs/datasets/ceval/ceval_zero_shot_gen_bd40ef.py |
+----------------------------+------------------------------------------------------+
```

#### (4)启动评测

确保按照上述步骤正确安装`OpenCompass`并准备好数据集后，可以通过以下命令评测`InternLM-Chat-7B`模型在`C-Eval`数据集上的性能。由于`OpenCompass`默认并行启动评估过程，我们可以在第一次运行时以`--debug`模式启动评估，并检查是否存在问题。在`--debug`模式下，任务将按顺序执行，并实时打印输出。

```sh
python run.py --datasets ceval_gen --hf-path /share/temp/model_repos/internlm-chat-7b/ --tokenizer-path /share/temp/model_repos/internlm-chat-7b/ --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 2048 --max-out-len 16 --batch-size 4 --num-gpus 1 --debug
```

命令解析：

```sh
--datasets ceval_gen \
--hf-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace 模型路径
--tokenizer-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 2048 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 4  \  # 批量大小
--num-gpus 1  # 运行模型所需的 GPU 数量
--debug
```

如果一切正常，您应该看到屏幕上显示 “Starting inference process”：

```sh
[2024-01-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

评测完成后，将会看到：

```sh

dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_model_repos_internlm-chat-7b
----------------------------------------------  ---------  -------------  ------  -------------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                         31.58
ceval-operating_system                          1c2571     accuracy       gen                                                                         36.84
ceval-computer_architecture                     a74dad     accuracy       gen                                                                         28.57
ceval-college_programming                       4ca32a     accuracy       gen                                                                         32.43
ceval-college_physics                           963fa8     accuracy       gen                                                                         26.32
ceval-college_chemistry                         e78857     accuracy       gen                                                                         16.67
ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                         21.05
ceval-probability_and_statistics                65e812     accuracy       gen                                                                         38.89
ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                         18.75
ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                         35.14
ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                         50
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                         22.22
ceval-high_school_physics                       adf25f     accuracy       gen                                                                         31.58
ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                         15.79
ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                         36.84
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                         26.32
ceval-middle_school_biology                     86817c     accuracy       gen                                                                         61.9
ceval-middle_school_physics                     8accf6     accuracy       gen                                                                         63.16
ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                         60
ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                         47.83
ceval-college_economics                         f3f4e6     accuracy       gen                                                                         41.82
ceval-business_administration                   c1614e     accuracy       gen                                                                         33.33
ceval-marxism                                   cf874c     accuracy       gen                                                                         68.42
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                         70.83
ceval-education_science                         591fee     accuracy       gen                                                                         58.62
ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                         70.45
ceval-high_school_politics                      5c0de2     accuracy       gen                                                                         26.32
ceval-high_school_geography                     865461     accuracy       gen                                                                         47.37
ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                         52.38
ceval-middle_school_geography                   8a63be     accuracy       gen                                                                         58.33
ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                         73.91
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                         63.16
ceval-logic                                     f5b022     accuracy       gen                                                                         31.82
ceval-law                                       a110a1     accuracy       gen                                                                         25
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                         30.43
ceval-art_studies                               2a1300     accuracy       gen                                                                         60.61
ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                         62.07
ceval-legal_professional                        ce8787     accuracy       gen                                                                         39.13
ceval-high_school_chinese                       315705     accuracy       gen                                                                         63.16
ceval-high_school_history                       7eb30a     accuracy       gen                                                                         70
ceval-middle_school_history                     48ab4a     accuracy       gen                                                                         59.09
ceval-civil_servant                             87d061     accuracy       gen                                                                         53.19
ceval-sports_science                            70f27b     accuracy       gen                                                                         52.63
ceval-plant_protection                          8941f9     accuracy       gen                                                                         59.09
ceval-basic_medicine                            c409d6     accuracy       gen                                                                         47.37
ceval-clinical_medicine                         49e82d     accuracy       gen                                                                         40.91
ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                         45.65
ceval-accountant                                002837     accuracy       gen                                                                         26.53
ceval-fire_engineer                             bc23f5     accuracy       gen                                                                         22.58
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                         64.52
ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                         34.69
ceval-physician                                 6e277d     accuracy       gen                                                                         40.82
ceval-stem                                      -          naive_average  gen                                                                         35.09
ceval-social-science                            -          naive_average  gen                                                                         52.79
ceval-humanities                                -          naive_average  gen                                                                         52.58
ceval-other                                     -          naive_average  gen                                                                         44.36
ceval-hard                                      -          naive_average  gen                                                                         23.91
ceval                                           -          naive_average  gen                                                                         44.16
```

有关`run.py`支持的所有与HuggingFace相关的参数，请阅读[评测任务发起](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/experimentation.html#id2)。

除了通过命令行配置实验外，`OpenCompass`还允许用户在配置文件中编写实验的完整配置，并通过`run.py`直接运行它。配置文件是以`Python`格式组织的，并且必须包括`datasets`和`models`字段。

```py
from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .models.opt.hf_opt_125m import opt125m
    from .models.opt.hf_opt_350m import opt350m

datasets = [*siqa_datasets, *winograd_datasets]
models = [opt125m, opt350m]
```

运行任务时，我们只需将配置文件的路径传递给`run.py`：

```sh
python run.py configs/eval_demo.py
```
