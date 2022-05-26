#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0 # 用于计算初始化范围

        # torch.nn.Parameter(Tensor data, bool requires_grad)将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数，并将这个参数绑定到module里面，成为module中可训练的参数
        # data为传入Tensor类型参数，requires_grad默认值为True，表示可训练，False表示不可训练
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), # 计算三元组的分数的边界值gamma,初始化范围的计算也用到    fixed margin
            requires_grad=False
        )

        # rotate: embedding_range：（24+2）/1000=0.026
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),  # 计算初始化范围
            requires_grad=False
        )

        # rotate: entity_dim 2000   relation_dim  1000
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim  # 实体维度是否乘2
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim  # 关系维度是否乘2
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        # entity_embedding服从由a到b的均匀分布,也就是U～(-0.026,0.026)  这里应该就是对实体进行初始化，但是不知道为什么这样初始化
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # 对关系进行初始化，relation_embedding服从由a到b的均匀分布，也就是U～(-0.026,0.026)
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]])) #  --------------------------------------------------------------------------
        # 选择模型
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
        # Rotate需要double entity embedding
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')
        # ComplEx要double_entity_embedding and double_relation_embedding
        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        计算一批三元组得分的正向函数。
        在“single”模式下，样品是一批三元组。在“head-batch”或“tail-batch”模式下，样品由两部分组成。第一部分通常是正样本。
        第二部分是负样本。因为负样本和正样本通常在三元组中共享两个元素（（head，relation）或（relation，tail））。
        '''

        if mode == 'single': # 默认mode是single，用于计算正样本的分数
            batch_size, negative_sample_size = sample.size(0), 1 # 这句没有用
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1) # (1024,1,1000)

            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1) # (1024,1,1000)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1) # (1024,1,1000)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample # head_part（1024，3）   tail_part（1024，256）
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1) # batch_size：1024   negative_sample_size：256
            # torch.index_select返回按照相应维度的给定index的选取的元素，index必须是longtensor https://pytorch.org/docs/stable/generated/torch.index_select.html
            head = torch.index_select(  # 根据头实体的编号拿出对应的向量
                self.entity_embedding,  # self.entity_embedding：（14951，1000）
                dim=0, 
                index=head_part[:, 0]  # index：（1024，）      head_part[:, 0]代表这个batch里面所有的头实体的编号，例如[2996,1233,3316,.....]
            ).unsqueeze(1)  # （第0维的大小与index的相同，其他的维度与inout的相同）  (1024,1000) >> (1024,1,1000)  代表1024个1000维的向量
            
            relation = torch.index_select( # 根据关系的编号拿出对应的向量
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)   # (1024,1000) >> (1024,1,1000)  代表1024个1000维的向量

            tail = torch.index_select(
                self.entity_embedding,  # （14951，1000）
                dim=0, 
                index=tail_part.view(-1)  # （262144,) 共有1024*256=262144个尾实体，把这些尾实体编号对应的向量都拿出来
            ).view(batch_size, negative_sample_size, -1)  # (262144,1000) >> (1024,256,1000)没看懂
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail) # (1024,1,1000)+(1024,1,1000)-(1024,256,1000)=(1024,256,1000)  test: (16,14951,1000)+(16,1,1000)-(16,1,1000)=(16,14951,1000)
        else:
            score = (head + relation) - tail # (1024,1,1000)+(1024,1,1000)-(1024,256,1000)=(1024,256,1000)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)  # {torch.norm：对输入的Tensor求p范数  (1024,256,1000)>>(1024,256)}  然后用gamma减去每（1024，256）的每一个值返回（1024，256）test: 返回(16,14951)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode): # head: (1024,1,2000) relation: (1024,1,2000) tail: (1024,256,2000)
        re_head, im_head = torch.chunk(head, 2, dim=2) # chunk：把head拆成两块,也就是分成实数和复数的部分 https://pytorch.org/docs/stable/generated/torch.chunk.html
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode): # RotatE模型实现
        pi = 3.14159265358979323846
        # head: (1024,1,2000)    tail: (1024,1,2000)     relation: (1024,256,1000)
        re_head, im_head = torch.chunk(head, 2, dim=2)  # 分块  实数域和复数域
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]  关系嵌入的相位在[-pi, pi]之间均匀初始化
        # 关系的复数通过欧拉方程实现   cos(θ) + isin(θ)  这样可以限定关系的模长为1
        phase_relation = relation/(self.embedding_range.item()/pi)  # 一个trick,目的应该是把实体和关系拉齐到同一级别（由于关系这里进行了cos/sin计算）
        re_relation = torch.cos(phase_relation) # cos(θ)
        im_relation = torch.sin(phase_relation) # sin(θ)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0) # score: tensor(2,1024,256,1000)
        score = score.norm(dim = 0) # 二范数    score: tensor(1024,256,1000)

        score = self.gamma.item() - score.sum(dim = 2) # score: tensor(1024,256)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        # Make phases of entities and relations uniformly distributed in [-pi, pi]
        # 这样可以限定头实体、尾实体、关系的模长为1
        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train() # 让你的模型知道现在正在训练。像dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。

        optimizer.zero_grad() # 每一轮batch需要设置optimizer.zero_grad，根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。
        # 取出正负样本       positive_sample:tensor:(1024,3) 这是batch个正确的三元组       negative_sample:tensor:(1024,256) 这是针对这batch个正确三元组的256个错误的尾/头实体
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda() # 正确的三元组
            negative_sample = negative_sample.cuda() # 对于每个正确的三元组的一组错误的尾实体
            subsampling_weight = subsampling_weight.cuda() # 在采样过程中以一定的概率丢弃一些？---------------------------------------------------------------------

        negative_score = model((positive_sample, negative_sample), mode=mode)  # (1024,256) 256个负样本的尾实体，针对每一个尾实体有一个负样本的分数

        if args.negative_adversarial_sampling: # 是否加自对抗的负采样策略，也就是对应：F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
            # 在自我对抗抽样中，我们不对抽样权重应用反向传播,也就是这部分(F.softmax(negative_score * args.adversarial_temperature, dim = 1)
            # 对应论文中公式（6）的右半部分， F.softmax(***)对应公式（5）
            # .detach():训练网络的时候可能希望保持一部分的网络参数不变, 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处是得到的这个tensor永远不需要计算其梯度，不具有grad
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)  # (1024,256)>>(1024,)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample) # (1024,1)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1) # (1024,) rotate中的loss计算方法

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            '''
            论文中没有这一部分    类似word2vec中的subsampling方法：https://www.cnblogs.com/TMatrix52/p/11976737.html
            subsampling_weight的计算：
            subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)] # 针对这一个三元组，计算关系和头实体在一起的次数加上关系和尾实体在一起的次数
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight])) # 开根号,计算一个subsampling_weight
            个人理解：因为某些三元组的(头关系),(关系尾)可能出现的频率很高，而另一些三元组(头关系),(关系尾)出现的频率很低，计算score时高频的影响会很大，存在不平衡的现象，低频的信息也应该被模型所注意
            为了减弱罕见三元组数据与高频三元组数据间存在不平衡现象，使用一个自定义的subsampling方法
            由subsampling_weight的公式可以看出，跟这个三元组有关的数据出现的频率越高，subsampling_weight越小，乘subsampling_weight后那么针对这个三元组的分数占所有分数的比例就越小
            减少了高频数据与低频数据之间的差距
            '''
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval() # 测试模式
        
        if args.countries: # 针对论文中的country数据集
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, # 所有的正确的三元组
                    args.nentity,  # 所有的实体个数
                    args.nrelation,  # 所有的关系个数
                    'head-batch'
                ), 
                batch_size=args.test_batch_size, # 16
                num_workers=max(1, args.cpu_num//2), # 5
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list]) # head-batch和tail-batch的样本数
            # train时:  positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias # 选择出来的三元组(head,relation,tail1),其他的(head,relation,tail2)也会分数比较高，对这些分数减去1,降低他们的排名，使其不会干扰对性能的判断

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True) # 分数越大越好 返回从大到小排序后的值所对应原a的下标，即torch.sort()返回的indices
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0] # 正确的头实体的编号
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2] # 正确的尾实体的编号
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero() # nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
                            assert ranking.size(0) == 1 # 如果ranking.size(0) == 1，程序正常往下运行

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            # 指标
                            logs.append({
                                'MRR': 1.0/ranking, # 越大越好
                                'MR': float(ranking), # MRR倒数
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs) # 对log的每一个metric都求平均值

        return metrics
