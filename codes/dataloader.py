#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples) # 获取部分三元类（head，relation）或（relation，tail）的频率
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples) # 统计(头实体，关系)的尾实体有什么以及(关系，尾实体)的头实体有什么
        
    def __len__(self): # 返回样本个数
        return self.len
    
    def __getitem__(self, idx):# 返回数据集和标签
        positive_sample = self.triples[idx] # 取出一个正样本
        # 针对这一个正样本制作负样本
        head, relation, tail = positive_sample
        # subsampling_weight：下采样权重
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)] # 针对这一个三元组，计算关系和头实体在一起的次数加上关系和尾实体在一起的次数
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight])) # 开根号,计算一个subsampling_weight
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size: # self.negative_sample_size：256
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2) # 随机生成256*2个实体的编号，至于需要256个为啥整512个呢，因为后续要去除正确的三元组，这里的目的是保证最后能剩256个
            # 从low到high（左闭右开）返回一个给定size的list，没有high的时候，返回从0到low的值，这里的low是entity的数量
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch': # 随机生成的数当成尾实体的编号
                # np.in1d  从b序列找跟a序列相同的值，返回true和false,构成这样一个向量，mask是512个布尔值
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)], # 从所有的正样本中寻找与我们构建的negative sample的相同的值，返回他们的true和false的结果
                    # 这样我们就把head  relation对应的tail的索引找到了，生成了mask，根据这个mask把随机生成的负样本中的正确三元组去掉
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask] # 生成了mask以后，把false的去掉，也就是去除正确的样本
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size] # 把获得的负样本拼接起来 https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html

        negative_sample = torch.LongTensor(negative_sample) # 转成longtensor,后面取向量的index得是longtensor | negative_sample:tensor:(256,) 这是针对这一个正确三元组的256个错误的尾/头实体

        positive_sample = torch.LongTensor(positive_sample) # 转成longtensor | positive_sample:tensor:(3,) 这是一个正确的三元组

        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data): # data={list:1024}  每一个元素是一个tuple:4，tuple的每一个元素分别是正确的三元组、错误的三元组的头/尾实体、下采样权重、mode
        positive_sample = torch.stack([_[0] for _ in data], dim=0) # tensor:(1024,3) 正确的三元组
        negative_sample = torch.stack([_[1] for _ in data], dim=0) # tensor:(1024,256) 错误的三元组的尾实体
        subsample_weight = torch.cat([_[2] for _ in data], dim=0) # tensor:(1024,)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4): # 对应self.count    为什么start是4？
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec # 统计用于下采样的频率
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples): # 统计(头实体，关系)的尾实体有什么以及(关系，尾实体)的头实体有什么，返回字典
        '''
        Build a dictionary of true triples that will    建立一个真三元组字典
        be used to filter these true triples for negative sampling   用于过滤负采样的真三元组
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail) # {头，关系: [尾的列表]}
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head) # {关系，尾: [头的列表]}
        # 把list转成ndarray的格式 {(123, 456): [112, 121, 111]}  >>>  {(123, 456): array([112, 121, 111])}
        for relation, tail in true_head: # 后面用到np.in1d，因此要转化成ndarray
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)]))) # 去重并且用np.array(list)方法来创建ndarray数组
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples) # 去重的所有的正确三元组
        self.triples = triples # 测试三元组
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode # head-batch/tail-batch

    def __len__(self):
        return self.len
    '''
    TestDataset中没有负样本，但是要从所有的头实体中进行选择排序
    [exp1 if condition else exp2 for x in data]
    此处if...else主要起赋值作用，当data中的数据满足if条件时将其做exp1处理，否则按照exp2处理，最后统一生成为一个数据列表
    遍历所有的实体，若不在所有的正确三元组里，则返回一个(0,rand_head)的元素，若在三元组里，则返回一个(-1, head)的元素，head是固定值，rand_head是遍历的元素
    head = 11267时
    会得到类似 tmp = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(-1,11267),(0,7)........(0,215),(-1,11267),......(0,11266),(-1,11267)(0,11268).......(0,14951)]
    再经过tmp[head] = (0, head) 
    最后会得到类似 tmp = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(-1,11267),(0,7)........(0,215),(-1,11267),......(0,11266),(0,11267)(0,11268).......(0,14951)]
    也就是11267位置的变成了正常按顺序排列的(0,rand_head)
    '''
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        # 此处if...else主要起赋值作用，当data中的数据满足if条件时将其做exp1处理，否则按照exp2处理，最后统一生成为一个数据列表
        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head) # 最终里面有两种类型的元素，分别是(0,rand_head)(-1,head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)  # tensor: (14951,2)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1] # tensor(14951,)

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader): # 产生一个生成器对象输出dataloader的数据：yield的函数则返回一个可迭代的 generator（生成器）对象，你可以使用for循环或者调用next()方法遍历生成器对象来提取结果。
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
