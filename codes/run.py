#!/usr/bin/python3
# from __future__ import作用是将新版本的特性引进当前版本中，也就是说我们可以在当前版本使用新版本的一些特性,例如想用python2.x体验python3.x的写法
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging # 日志功能
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration 覆盖模型和数据设置
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console  将日志写入检查点和控制台
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log') # 定义训练日志文件名，os.path.join()函数：路径拼接
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')  # 定义测试日志文件名
    # logging由logger，handler，filter，formater四个部分组成
    # logger是提供我们记录日志的方法；handler是让我们选择日志的输出地方，如：控制台，文件，邮件发送等，一个logger可以添加多个handler；
    # filter是给用户提供更加细粒度的控制日志的输出内容；formater用户格式化输出日志的信息
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s', # 初始化了日志输出的格式，后面代码中只需要写内容(对应message)，时间等信息都会自动带上
        level=logging.INFO,  # 当我们指定一个输出级别后，只有该级别和级别更高（数值更大）的日志会输出，每个级别都对应一个输出函数
        datefmt='%Y-%m-%d %H:%M:%S',  # datefmt对时间格式进行修改
        filename=log_file, # 日志文件名字
        filemode='w'  # 写入
    )
    console = logging.StreamHandler() # 创建一个handler，用于输出到控制台
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s') # 定义handler的输出格式（formatter）
    console.setFormatter(formatter) # 给handler添加formatter
    logging.getLogger('').addHandler(console) # 用logging.getLogger(name)方法进行初始化，name可以不填；给logger添加handler,handler可以理解为规则

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test): # 提示需要选择训练模式
        raise ValueError('one of train/val/test mode must be choosed.') # raise ValueError 自定义的异常处理

    # 覆盖模型和数据设置，暂时不懂干啥的
    if args.init_checkpoint:  # 默认为None, parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
        override_config(args)
    elif args.data_path is None: # 检查是否有数据可以读取
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None: # 检查是否有路径用于保存模型
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path): # 检查并创建保存路径
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args) # 创建并写入日志文件
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin: # 把实体赋值编码
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')  # '0	/m/014_24'
            entity2id[entity] = int(eid)  # 格式 {'/m/014_24': 0}

    with open(os.path.join(args.data_path, 'relations.dict')) as fin: # # 把关系赋值编码
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries: # 如果使用countries这个数据集
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id) # 实体个数
    nrelation = len(relation2id)  # 关系个数
    
    args.nentity = nentity # 赋值到全局变量args.nentity
    args.nrelation = nrelation # 赋值到全局变量args.nrelation
    
    logging.info('Model: %s' % args.model)  #  2022-04-29 18:21:41 INFO     Model: RotatE
    logging.info('Data Path: %s' % args.data_path)  # 2022-04-29 18:21:45 INFO     Data Path: ../data/FB15kzu
    logging.info('#entity: %d' % nentity)  #  2022-04-29 18:21:46 INFO     #entity: 14951
    logging.info('#relation: %d' % nrelation)  #  2022-04-29 18:21:47 INFO     #relation: 1345
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id) # 读取训练集的三元组编号
    logging.info('#train: %d' % len(train_triples)) # 返回三元组的数字
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id) # 读取验证集的三元组编号
    logging.info('#valid: %d' % len(valid_triples)) # 返回三元组的数字
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id) # 读取测试集的三元组编号
    logging.info('#test: %d' % len(test_triples))  # 返回三元组的数字
    
    #All true triples  测试的时候使用
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    # 下面的代码会在日志输出如下内容,也就是输出模型的可训练参数并且查看
    # 2022-04-30 09:15:21 INFO     Model Parameter Configuration:
    # 2022-04-30 09:18:50 INFO     Parameter gamma: torch.Size([1]), require_grad = False
    # 2022-04-30 09:18:52 INFO     Parameter embedding_range: torch.Size([1]), require_grad = False
    # 2022-04-30 09:18:53 INFO     Parameter entity_embedding: torch.Size([14951, 2000]), require_grad = True
    # 2022-04-30 09:18:54 INFO     Parameter relation_embedding: torch.Size([1345, 1000]), require_grad = True
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():  # named_parameters()返回各层中参数名称和数据
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda() # 使用GPU

    if args.do_train:
        # Set training dataloader iterator
        # 设置训练集头实体的dataloader        头实体，目前这里头和尾定义是完全一样的
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),  # 这是定义的dataset类
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn  # collate_fn：表示合并样本列表以形成小批量的Tensor对象,如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        )
        # 尾实体
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), # 这是定义的dataset类
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)  #
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam( # filter(函数，序列)函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps  # warmup步长阈值,即train_steps<warmup_steps,使用预热学习率,否则使用预设值学习率
        else:
            warm_up_steps = args.max_steps // 2 # 如果没设置warm up参数，就将warm up参数设置到最大步数的一半

    if args.init_checkpoint: # 用来还原训练好的模型？
        # Restore model from checkpoint directory  从检查点目录还原模型
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint')) # 用来加载torch.save() 保存的模型文件
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train: # 读取模型后是否还继续训练
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)  # 2022-04-30 13:53:03 INFO     Ramdomly Initializing RotatE Model...
        init_step = 0
    
    step = init_step # 初始化步数
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:  # 是否使用自对抗负采样
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    # 2022-04-30 14:02:37 INFO     Start Training...
    # 2022-04-30 14:02:42 INFO     init_step = 0
    # 2022-04-30 14:02:42 INFO     batch_size = 1024
    # 2022-04-30 14:02:43 INFO     negative_adversarial_sampling = 1
    # 2022-04-30 14:02:43 INFO     hidden_dim = 1000
    # 2022-04-30 14:02:43 INFO     gamma = 24.000000
    # 2022-04-30 14:03:55 INFO     negative_adversarial_sampling = True
    # 2022-04-30 14:07:28 INFO     adversarial_temperature = 1.000000
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps): # 开始训练
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0: # 每10000步保存一次模型
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0: # 每100步保存一次log, 把log信息存在这里，然后training_log置空，为下一轮作准备
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0: # 每10000步验证一下
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
