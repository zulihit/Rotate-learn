
# RotatE模型的pytorch实现

**简介**

本仓库为个人学习所用，记录了本人对于知识图谱嵌入的Rotate模型的学习过程

本代码基于pytorch实现，代码格式非常正规，适合初学者学习

原作者代码：

https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

论文链接：

https://openreview.net/forum?id=HkgEQnRqYQ

**数据集**

关于数据集部分，在原作者仓库中下载，复制到主目录中即可，没有的文件夹可能需要创建一下

![image](https://user-images.githubusercontent.com/68625084/170436074-c3b3dcc2-1b18-4154-a1e5-f7d68cbce48a.png)


**实现的模型**

Models:
 - [x] RotatE
 - [x] pRotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult

评价指标:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**数据集格式**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Train**

For example, this command train a RotatE model on FB15k dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k \
 --model RotatE \
 -n 256 -b 1024 -d 1000 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save models/RotatE_FB15k_0 --test_batch_size 16 -de
```

在linux终端输入上述代码即可运行程序，在GPU：0训练RotatE模型

**Test**

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

**Reproducing the best results**

To reprocude the results in the ICLR 2019 paper [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ), you can run the bash commands in best_config.sh to get the best performance of RotatE, TransE, and ComplEx on five widely used datasets (FB15k, FB15k-237, wn18, wn18rr, Countries).

The run.sh script provides an easy way to search hyper-parameters:

    bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 200000 16 -de
    
这里是使用bash脚本来运行程序，更加的方便，对于实现的模型最佳超参数都在best_config.sh文件里

需要哪个复制出来在linux终端里面运行即可

**Speed**

The KGE models usually take about half an hour to run 10000 steps on a single GeForce GTX 1080 Ti GPU with default configuration. And these models need different max_steps to converge on different data sets:

| Dataset | FB15k | FB15k-237 | wn18 | wn18rr | Countries S* |
|-------------|-------------|-------------|-------------|-------------|-------------|
|MAX_STEPS| 150000 | 100000 | 80000 | 80000 | 40000 | 
|TIME| 9 h | 6 h | 4 h | 4 h | 2 h | 

这是原作者根据不同模型的训练时间

**Results of the RotatE model**

| Dataset | FB15k | FB15k-237 | wn18 | wn18rr |
|-------------|-------------|-------------|-------------|-------------|
| MRR | .797 ± .001 | .337 ± .001 | .949 ± .000 |.477 ± .001
| MR | 40 | 177 | 309 | 3340 |
| HITS@1 | .746 | .241 | .944 | .428 |
| HITS@3 | .830 | .375 | .952 | .492 |
| HITS@10 | .884 | .533 | .959 | .571 |

模型运行的结果

**更加基础的TransE代码**

https://github.com/zulihit/transE-learn

