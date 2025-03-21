# DRL IMPLEMENTATION
An implementation of the Deep Q-Network (DQN) and Double Deep Q-Network(D2QN) and D3QN.
![img](https://gitee.com/abstcol/imagebed/raw/master/20250304190112632.gif)![](https://raw.githubusercontent.com/abstcol/imagebed/main/20250309145034149.gif)


## Project Directory Structure

gym\
├── agent_algo\
│   &emsp;&emsp;&emsp;&emsp;  ├── D2QN.py\
│   &emsp;&emsp;&emsp;&emsp;  ├── D3QN.py\
│   &emsp;&emsp;&emsp;&emsp;  ├── DQN.py\
│   &emsp;&emsp;&emsp;&emsp;  ├── __init__.py\
│   &emsp;&emsp;&emsp;&emsp;  └── utils.py\
├── argument.py\
├── env.py\
├── README.md\
├── requirements.txt\
└── run_rl\
 &emsp;&emsp;&emsp;&emsp; ├── no_train_car.py\
 &emsp;&emsp;&emsp;&emsp; ├── test.py\
 &emsp;&emsp;&emsp;&emsp; ├── test_utils.py\
 &emsp;&emsp;&emsp;&emsp; ├── train.py\
 &emsp;&emsp;&emsp;&emsp; └── train_utils.py\
 
**`agent_algo`**: Contains agent classes defined by various reinforcement learning algorithms. Each file (e.g., `DQN.py`, `D2QN.py`, etc.) implements a specific algorithm. The `utils.py` file includes helper functions related to the algorithms.\
**`argument.py`**: Defines all the command-line arguments used in the program.\
**`env.py`**: Contains wrapper classes that override the environment's `step` function to interact with agents.\
**`run_rl`**: Includes the training and testing scripts. Helper functions for training and testing are in `train_utils.py` and `test_utils.py`. During execution, the directory also generates files related to the run (such as `wandb`, `experiments`, `gif`, `artifacts`)

### Additional Information

-   **`requirements.txt`**: Lists all the dependencies required by the project.
-   **`README.md`**: Provides documentation for the project.


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
cd run_rl
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
    torch.save(agent.actor.state_dict(), checkpoint_path)
    wandb.log_model(checkpoint_path, f"{args.env_name}_{args.algo_name}_{timestamp}",
                    [f"average_length_{sum([env.length_queue[episode - i] for i in range(50)]) / 50}"])

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
eyJoaXN0b3J5IjpbODcyMzAzMDk2LC0xMzM1NTkzNjI0LC05Nj
E1NzUzOTMsLTE3ODUxMzI1MDQsMTQxNjA5NjQwOSwxMzU3MTEx
MjAzLDExNjQ1NDM3NzRdfQ==
-->