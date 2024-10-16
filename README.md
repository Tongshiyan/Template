这是一个深度学习的代码框架，可以用作模板，使用此模板时请点星标
-

采用VGG16网络进行光伏电池PV的缺陷检测（分类任务）

数据集：https://github.com/zae-bayern/elpv-dataset

注：labels.csv经过分列与添加标题行的处理


runs.py 运行文件
-

test.py 调试 
-

experiment 实验配置文件夹
   -
     configs.py 实验配置，添加删除某些常量
   
     EXP.py 实验流程代码

models 模型文件夹
   -
     model.py 主要模型 ours

util 常用文件夹
   -
     libraries.py 常用库
     metrics.py 评价指标
     tools.py 提前跳出循环工具

output
   -
    输出保存为checkpoints文件夹 其中记录了某次实验的最优模型参数
    测试结果保存至test_results文件夹
    
This is a deep learning code framework that can be used as a template, please star when using this template
-

Defect detection in PV cells PV using VGG16 network (classification task)

Dataset: https://github.com/zae-bayern/elpv-dataset

Note: labels.csv is disaggregated and header rows are added

runs.py run file
-

test.py debugging 
-

experiment Experiment configuration folder
   -
    configs.py Experiment configuration, adding and removing some constants
    
    EXP.py experiment flow code

models models folder
   -
    model.py Main model ours

util common folder
   -
     libraries.py Common libraries
     
     metrics.py Evaluation metrics
     
     tools.py Tools to jump out of loops early

output
   -
    The output is saved as a checkpoints folder.
    
    Output is saved to the checkpoints folder, which records the optimal model parameters for a given experiment.
    
    Test results are saved to the test_results folder