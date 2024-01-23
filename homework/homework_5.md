# 【Lectrue-5】课后作业

## 基础作业
* 使用 LMDeploy 以本地对话、网页Gradio、API服务中的一种方式部署 InternLM-Chat-7B 模型，生成 300 字的小故事（需截图）

![](../attach/homework_5_1.jpg)

![](../attach/homework_5_2.jpg)

## 进阶作业
* 将第四节课自我认知小助手模型使用LMDeploy量化部署到OpenXLab平台。
* 对InternLM-chat-7B模型进行量化，并同时使用KV Cache量化，使用量化后的模型完成API服务的部署，分别对比模型量化前后（将bs设置为1和maxlen设置为512）和KV Cache量化前后（将bs设置为（将bs设置为8和maxlen设置为2048）的显存大小。
* 在自己的任务数据集上任取若干条进行BenchMark测试，测试方向包括：
    * (1)TurboMin推理+Python代码集成
    * (2)在(1)的基础上采用W4A16量化
    * (3)在(1)的基础上开启KV Cache量化
    * (4)在(2)的基础上开启KV Cache量化
    * (5)使用Hugging Face推理

*开发中，将与“整体实训营项目”一起展示。*