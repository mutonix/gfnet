### Primary Exam in Intelligent Computing Lab 

#### Problems solved (1+2+4):
- Task 1: Defect Classification
- Task 2: Transfer learning on defect classification
- Task 4: Reimplementation on GFNet

****
**This repository is contributed by Team 3: 杨东杰(mainly) 郭清妍 金孟群.**

****
### How to use (mainly for GFNet)



#### Task 1 Defect Classification
##### Train a model to classify defects
- Download the dataset in Task 1 and put in under the './data' folder.
  
  <https://pan.baidu.com/s/1DBKlSWRFT-TMMoJgAYAZ3AD>  
  code: ehwl
<br >

- Choose a model to train. (Res2Net or GFNet)
    ```
    pip install -r requirements.txt
    python train.py --task_name task0 --model gfnet --batch_size 64 --eval_batch_size 64 --epochs 10 --lr 0.001 --eval_period 1 --lr_decay_step 3 --drop_rate 0
    ```
    **Parameters Explanation**
    --task name : name for the current training task.
    --model : choices for model type [gfnet] / [res2net]. 
    --eval_period : evaluate the model using testset every [eval_period].

##### Training Process & Results
- checkpoint with best accuracy on testset will be saved.
- tensorboard supported.
- a txt file will also record the loss, accuracy on trainset & testset.

#### Task 2: Transfer Learning
##### Training from scratch
- Download the NEU dataset(prepocessed) and put it under the './data' folder
    <https://pan.baidu.com/s/1S9k3pzvrI17UT7GKsojdNQ>
    code :ydj6
    **1584 for train  &  216 for test**
    <br >
    The structure of files in './data' folder:
    ```
    --data
        --origin
            --train
            --val
        --NEU
            --train
            --val
    ```
- Training from scratch on NEU Dataset
    ```
    python train.py --task_name task0 --model gfnet --batch_size 64 --eval_batch_size 64 --epochs 10 --lr 0.001 --eval_period 1  --lr_decay_step 3  --data_path ./data/NEU --drop_rate 0
    ```
    --data_path : the path of the dataset. [Default: path of dataset in Task 1]

##### Transfer Learning
- Download the pre-trained weight in Task 1
    <https://pan.baidu.com/s/1PIB_YMcglAGb2aKBVXeWWQ>
    code : ydj6
    <br >
- Training on NEU dataset using transfer learning  
    ```
    python train.py --task_name task0 --model gfnet --batch_size 64 --eval_batch_size 64 --epochs 10 --lr 0.001 --eval_period 10 --lr_decay_step 5 --transfer ./checkpoint/GFNet_origin(no_drop).pth --data_path ./data/NEU --drop_rate 0
    ```
    --transfer : the path of the pre-trained weight. [Default: '', not using transfer learning]
    
#### Task 4: Reimplementation on GFNet

- Details can be seen in **gfnet.py**.

### Our results
Our results are shown on the PPT.

