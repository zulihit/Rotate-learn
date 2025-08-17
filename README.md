# RotatE

This repository contains a personal PyTorch implementation of the **RotatE** model, designed for learning and practicing knowledge graph embedding (KGE) methods.  

---

## Introduction
- This repo documents the process of learning RotatE for knowledge graph embedding.  
- The implementation follows clean and standardized PyTorch practices, making it suitable for beginners.  
- For detailed explanations, please refer to the following resources:  
  - Zhihu article: [https://zhuanlan.zhihu.com/p/520284915](https://zhuanlan.zhihu.com/p/520284915)  
  - Original codebase: [https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)  
  - Paper: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ)  

---

## Datasets
- Datasets can be downloaded from the original repository and copied into the root directory.  
- Create missing folders if necessary.  

Example dataset folder structure:  
![dataset-structure](https://user-images.githubusercontent.com/68625084/170436074-c3b3dcc2-1b18-4154-a1e5-f7d68cbce48a.png)  

---

## Implemented Models
- [x] RotatE  
- [x] pRotatE  
- [x] TransE  
- [x] ComplEx  
- [x] DistMult  

**Evaluation Metrics:**  
- [x] MRR, MR, Hits@1, Hits@3, Hits@10 (filtered)  
- [x] AUC-PR (for *Countries* datasets)  

**Loss Functions:**  
- [x] Uniform Negative Sampling  
- [x] Self-Adversarial Negative Sampling  

---

## Dataset Format
Knowledge graph data should include the following files:  
- `entities.dict`: maps entities to unique IDs  
- `relations.dict`: maps relations to unique IDs  
- `train.txt`: training set of triples  
- `valid.txt`: validation set (create a blank file if not available)  
- `test.txt`: test set for evaluation  

---

## Training
Example: Training a RotatE model on **FB15k** using GPU 0:  

```bash
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train  --cuda  --do_valid  --do_test  --data_path data/FB15k  --model RotatE  -n 256 -b 1024 -d 1000  -g 24.0 -a 1.0 -adv  -lr 0.0001 --max_steps 150000  -save models/RotatE_FB15k_0 --test_batch_size 16 -de
```

Run this command in a Linux terminal to train the RotatE model on GPU 0.  

---

## Testing
```bash
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE
```

---

## Reproducing the Best Results
To reproduce the results reported in the ICLR 2019 paper, use the `best_config.sh` script for RotatE, TransE, and ComplEx on five benchmark datasets (**FB15k, FB15k-237, wn18, wn18rr, Countries**).  

Example:  
```bash
bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 200000 16 -de
```

All optimal hyper-parameters are provided in `best_config.sh`. Copy the corresponding command and run it in a Linux terminal.  

---

## Training Speed
Approximate runtime on a single **GeForce GTX 1080 Ti GPU** (default configuration):  

| Dataset   | FB15k   | FB15k-237 | wn18   | wn18rr | Countries S* |
|-----------|---------|-----------|--------|--------|--------------|
| MAX_STEPS | 150000  | 100000    | 80000  | 80000  | 40000        |
| TIME      | 9 h     | 6 h       | 4 h    | 4 h    | 2 h          |  

---

## Results of RotatE
| Dataset | FB15k  | FB15k-237 | wn18  | wn18rr |
|---------|--------|-----------|-------|--------|
| MRR     | .797 ± .001 | .337 ± .001 | .949 ± .000 | .477 ± .001 |
| MR      | 40     | 177       | 309   | 3340   |
| Hits@1  | .746   | .241      | .944  | .428   |
| Hits@3  | .830   | .375      | .952  | .492   |
| Hits@10 | .884   | .533      | .959  | .571   |  

---

## Related Work
For a more basic implementation of TransE, see:  
- [https://github.com/zulihit/transE-learn](https://github.com/zulihit/transE-learn)
