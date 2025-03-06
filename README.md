# CartPole-v1
An implementation of Deep Q-Network (DQN) and Double Deep Q-Network(D2QN) in the CartPole-v1 environment.
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
conda create -n gym python=3.10  
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
`gamma`: Discount factor\
`algorithm`: DQN or D2QN

## Logs&Checkpoints

Training logs and checkpoints will be saved in wandb after experiement.
So please make sure that you log in your wandb account before experiement.
The wandb project and name is set as :
```python
# 初始化 wandb
    wandb.init(
        project="RL-Training",
        config=vars(args),  # 记录所有超参数
        name=f"{args.env_name}{args.algo_name}-{timestamp}"
    )
```
And the checkpoint will be uploaded every 50 epochs in this formate:
```python
# episode结束后的处理
        if (episode + 1) % 50 == 0:  # 每隔50个episode保存
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode + 1}.pth")
            torch.save(agent.main_q.state_dict(), checkpoint_path)
            wandb.log_model(checkpoint_path,f"{args.env_name }_{args.algo_name}_{timestamp}",[f"average_length_{sum([env.length_queue[episode-i] for i in range(50)])/50}"])

```




## Testing
After training the agent, you can test it using `test.py`:
```bash
python test.py --checkpoint-name {your-checkpoint-name-in-wandb}
```
Replace `{your-checkpoint-name-in-wandb}` with your actual checkpoint name.

##  Produce GIF
You can create a GIF of the agent's performance using `test.py`. Simply run the following command:
```bash
python test.py --checkpoint-name {your-checkpoint-name-in-wandb} --produce-gif
```
This will generate a GIF showcasing the agent's behavior during testing. Make sure to replace `{your-checkpoint-name-in-wandb}` with your actual checkpoint name.








<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMzU1OTM2MjQsLTk2MTU3NTM5MywtMT
c4NTEzMjUwNCwxNDE2MDk2NDA5LDEzNTcxMTEyMDMsMTE2NDU0
Mzc3NF19
-->