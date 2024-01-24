# 【Lectrue-5】课后作业

## 基础作业
* 使用 OpenCompass 评测 InternLM2-Chat-7B 模型在 C-Eval 数据集上的性能

### 1.下载InternLM2-Chat-7B模型

```sh
openxlab model download --model-repo 'OpenLMLab/internlm2-chat-7b'
```

### 2.下载并安装OpenCompass

```sh
git clone https://gitee.com/open-compass/opencompass.git
cd opencompass
pip install -e .
```

### 3.评测

```sh
python run.py \
    --datasets ceval_gen \
    --hf-path /data/stu/anhongjun/model/internlm2-chat-7b/ \
    --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
    --model-kwargs trust_remote_code=True device_map='auto' \
    --max-seq-len 2048 \
    --max-out-len 16 \
    --batch-size 4 \
    --num-gpus 1 \
    --debug
```

## 进阶作业
* 使用 OpenCompass 评测 InternLM2-Chat-7B 模型使用 LMDeploy 0.2.0 部署后在 C-Eval 数据集上的性能