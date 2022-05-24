#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes  # 代码地址
DATA_PATH=data   # 数据地址
SAVE_PATH=models # 保存模型地址

# bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 200000 16 -de
#The first four parameters must be provided
MODE=$1       # 代表传入的第1个参数   train
MODEL=$2      # 代表传入的第2个参数   RotatE
DATASET=$3    # 代表传入的第3个参数   FB15k
GPU_DEVICE=$4 # 代表传入的第4个参数   0
SAVE_ID=$5    # 代表传入的第5个参数   0

# 使用一个定义过的变量，只要在变量名前面加美元符号即可

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$6           # 代表传入的第6个参数    1024
NEGATIVE_SAMPLE_SIZE=$7 # 代表传入的第7个参数    256
HIDDEN_DIM=$8           # 代表传入的第8个参数    1000
GAMMA=$9                # 代表传入的第9个参数    24.0
ALPHA=${10}             # 代表传入的第10个参数   1.0
LEARNING_RATE=${11}     # 代表传入的第11个参数   0.0001
MAX_STEPS=${12}         # 代表传入的第12个参数   200000
TEST_BATCH_SIZE=${13}   # 代表传入的第13个参数   16

if [ $MODE == "train" ]; then

  echo "Start Training......"

# 通过bash脚本设置CUDA_VISIBLE_DEVICES,这句话一般在run.sh中开头，显示表示要使用的GPU
  CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    ${14} ${15} ${16} ${17} ${18} ${19} ${20}

elif [ $MODE == "valid" ]; then

  echo "Start Evaluation on Valid Data Set......"

  CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE

elif [ $MODE == "test" ]; then

  echo "Start Evaluation on Test Data Set......"

  CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

else
  echo "Unknown MODE" $MODE
fi
