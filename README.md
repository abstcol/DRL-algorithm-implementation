# CartPole-v1
An implementation of Deep Q-Network (DQN) in the CartPole-v1 environment.
![img](https://gitee.com/abstcol/imagebed/raw/master/20250304190112632.gif)

## Clone Repository 
First, clone the repository to your local machine: 
```bash 
git clone https://github.com/abstcol/gym.git 
cd gym
```

## Installation
Set up the project environment with:

```bash
conda create -n gym python=3.8  
conda activate gym 
pip install -r requirements.txt
```
If you encounter package conflicts, try installing dependencies manually.

## Training
You can train the agent with 
```bash
python train.py
```

You can customize the training process with various arguments.  
See `arguments.py` for more details, \
including:\
`lr`: Learning rate for the optimizer\
`batch_size`: Batch_size for training\
`gamma`: Discount factor

## Logs&Checkpoints

Training logs and checkpoints will be saved in:`logs/env{env you choose}-yyyymmdd-hhmmss` after experiment.
For example
```bash
logs/env-CartPole-v1-20250304-120000/
```

To visualize logs in TensorBoard, use:
```
tensorboard --logdir=logs
```



## Testing
After training the agent, you can test it using `test.py`:
```bash
python test.py --weight-path {w_path}
```
Replace `{w_path}` with your actual weight file path.

##  Produce GIF
You can create a GIF of the agent's performance using `test.py`. Simply run the following command:
```bash
python test.py --weight-path {w_path} --produce-gif
```
This will generate a GIF showcasing the agent's behavior during testing. Make sure to replace `{w_path}` with the actual path to your trained model's weights.








<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2ODY4MTAyNzYsLTE3ODUxMzI1MDQsMT
QxNjA5NjQwOSwxMzU3MTExMjAzLDExNjQ1NDM3NzRdfQ==
-->