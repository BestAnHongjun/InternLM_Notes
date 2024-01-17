# 【Lecture 4】XTuner 大模型单卡低成本微调实战

未完成，学习更新中...


## 一、Finetune的简介

LLM的下游应用中，**增量预训练**和**指令跟随**是经常会用到的两种微调模式。

### 1.增量预训练微调
* 使用场景：让基座模型学习到一些新知识，如某个垂直领域的常识
* 训练数据：文章、书籍、代码等

增量预训练的数据都是陈述句，没有问答的形式。训练时将“User”和“System”留空。

#### (1)LoRA微调
> Low Rank Adaptation of Large Language Models

* LLM的参数量主要集中在模型中的Linear，训练这些参数会耗费大量的显存
* LoRA通过在原本的Linera旁，新增一个支路，包含两个连续的小Linear，新增的这个支路通常叫做Adapter
* Adapter参数量远小于原本的Linear，能大幅降低训练的显存消耗

![](../attach/lecture_4_4.jpg)

#### (2)QLoRA微调

![](../attach/lecture_4_5.jpg)

### 2.指令跟随微调
* 使用场景：让模型学会对话模板，根据人类指令进行对话
* 训练数据：高质量的对话、问答数据

![](../attach/lecture_4_1.jpg)

#### (1)对话模板

在实际对话时，通常会有三种角色。
* System：给定一些上下文信息，比如“*你是一个安全的AI助手*”
* User：实际用户，会提出一些问题，比如“*世界第一高峰是？*”
* Assistant：根据User的输入，结合System的上下文信息，做出回答，比如“*珠穆朗玛峰*”

![](../attach/lecture_4_2.jpg)

对话模板是为了让LLM区分出System、User和Assistant，不同的模型会有不同的模板。

#### (2)XTuner工作流程

![](../attach/lecture_4_3.jpg)

## 二、XTuner介绍

### 1.功能亮点

* 适配多种生态
    * 多种微调策略与算法，覆盖各类SFT场景
    * 适配多种开源生态，如加载HuggingFace、ModelScope模型或数据集
    * 自动优化加速，开发者无需关注复杂的显存优化与计算加速细节
* 适配多种硬件
    * 训练方案覆盖NIVIDIA 20系以上所有显卡
    * 最低只需8G显存即可微调7B模型

### 2.快速上手

#### (1)安装

```sh
pip install xtuner
```

#### (2)挑选配置模板

```sh
xtuner list-cfg -p internlm_20b
```

#### (3)一键训练

```sh
xtuner  train interlm_20b_qlora_oasst1_512_e3
```

* Config命名规则

|项目|示例|备注|
|:-:|:-:|:-:|
|模型名|internlm_20b|无Chat代表是基座模型|
|使用算法|qlora||
|数据集|oasst1||
|数据长度|512||
|Epoch|e3,epoch3||

### 3.自定义训练

#### (1)拷贝配置模板

```sh
xtuner copy-cfg internlm_20b_qlora_oasst1_512_e3 ./
```

#### (2)修改配置模板

```sh
vim internlm_20b_qlora_oasst1_512_e3_copy.py
```

#### (3)启动训练

```sh
xtuner train internlm_20b_qlora_oasst1_512_e3_copy.py
```

* 常用超参数

|项目|备注|
|:-:|:-:|
|data_path|数据路径或HuggingFace仓库名|
|max_length|单条数据最大Token数，超过则截断|
|pack_to_max_length|是否将多条短数据拼接到max_length，提高GPU利用率|
|accumulative_counts|梯度累计，每多少次backward更新一次参数|
|evaluation_inputs|训练过程中，会根据给定的问题进行推理，便于观测训练状态|
|evaluation_freq|Evaluation的评测间隔iter数|

#### (4)一键对话接口

为了便于开发者查看训练效果，Xtuner提供了一键对话接口

* Float16模型对话

```sh
xtuner chat internlm/internlm-chat-20b
```

* 4bit模型对话

```sh
xtuner chat internlm/internlm-chat-20b --bits 4
```

* 加载Adapter模型对话

```sh
xtuner chat internlm/internlm-chat-20b --adapater $ADAPTER_DIR
```

## 三、8GB显卡玩转LLM
## 四、动手实战环节