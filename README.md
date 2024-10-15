这是一个深度学习的代码框架，可以用作模板
-

采用VGG16网络进行光伏电池PV的缺陷检测（分类任务）

数据集：https://github.com/zae-bayern/elpv-dataset


runs.py 运行文件
-

test.py 调试 
-

---experiment 实验配置文件夹
   -
   configs.py 实验配置，添加删除某些常量
   
   EXP.py 实验流程代码

---models 模型文件夹
   -
   model 主要模型 ours

---util 常用文件夹
   -
   libraries.py 常用库
   metrics.py 评价指标
   tools.py 提前跳出循环工具
