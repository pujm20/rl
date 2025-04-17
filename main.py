import numpy as np
import gym
import argparse
import torch
import matplotlib.pyplot as plt
import random
import seaborn as sns
import os
from PPO import Normalization, PPO_agent

def smooth(data, weight=0.9):
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(args, rewards, tag = 'train'):
    sns.set_theme()
    plt.figure()
    plt.title(f"{tag}ing curve on {args['device']} of {args['algo_name']} for {args['env_name']}")#内外引号需不同
    plt.xlabel('episodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    save_dir = os.path.join("results", args['env_name'], args['algo_name'])
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{tag}_rewards.png")
    plt.savefig(filename)
    #print(f"图像已保存到: {filename}")
    plt.show()


def all_seeds(env, seed = 2):
    '''
    设置随机种子
    '''
    #env.seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)#设置cpu随机种子
    torch.cuda.manual_seed_all(seed)#设置所有gpu随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)#设置python随机种子
    torch.backends.cudnn.deterministic = True#设置cudnn确定性算法
    torch.backends.cudnn.benchmark = False#设置cudnn基准算法
    torch.backends.cudnn.enabled = False#设置cudnn算法为False

def train(args, env, agent, state_dim):
    '''
    训练！
    '''
    print("开始训练！")
    rewards_list = []#记录所有回合总奖励的列表
    steps = []#记录所有回合迭代步数

    to_update_count = 0

    state_norm = Normalization(shape=state_dim)#状态归一化

    for i_ep in range(args['train_eps']):
        ep_reward = 0#该回合总奖励
        ep_step = 0#该回合总步数
        state, _ = env.reset()
        for _ in range(args['ep_max_steps']):
            ep_step += 1
            state = state_norm(state)#归一化状态trick
            action, log_prob = agent.take_action(state)
            #print(f"action: {action}, log_prob: {log_prob}")
            next_state, reward, dw, done, _ = env.step(action)#done表示智能体交互（死亡，达到目标等）结束；dw表示环境限制（最大步数/时间）结束与智能体交互结束
            #print(f"next_state: {next_state}, reward: {reward}, done: {done}, dw: {dw}")
            
            agent.memory.add((state, action, reward, next_state, dw, done, log_prob))#组成元组，存入经验回放池
            state = next_state
            to_update_count += 1
            # 确保有足够的样本数据再进行更新，避免标准差计算问题
            if to_update_count % args['update_rate'] == 0 :
                agent.update()
            ep_reward += reward
            #print(ep_reward)
            if done or dw:
                break
        steps.append(ep_step)
        rewards_list.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            # 确保打印信息并立即刷新输出缓冲区
            print(f"episode: {i_ep+1}/{args['train_eps']}, steps: {ep_step}, ep_reward: {ep_reward:.2f}")
            #env.render()
            
    print("完成训练！")
    plot_rewards(args, rewards_list, tag='train')
    env.close()

def test(args, env, agent, state_dim):
    print("开始测试！")
    rewards_list = []  # 记录所有回合的奖励
    steps = []
    state_norm = Normalization(shape=state_dim)#状态归一化
    for i_ep in range(args['test_eps']):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态;#state为元组，包含多个值；新版gym初始化返回为state, info；
        if isinstance(state, tuple):
            state = state[0]
        for _ in range(args['ep_max_steps']):
            ep_step+=1
            state = state_norm(state, update=False)
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done, dw, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done or dw:
                break
        steps.append(ep_step)
        rewards_list.append(ep_reward)
        print(f"episode：{i_ep+1}/{args['test_eps']}，reward：{ep_reward:.2f}")
    print("完成测试")
    plot_rewards(args, rewards_list, tag='test')
    
    env.close()

def main():
    #s = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='hyperparameters')
    parser.add_argument('--algo_name',default='PPO',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='MountainCarContinuous-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=1000,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--ep_max_steps',default = 10000,type=int,help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--epochs',default=10,type=int,help="eopchs of training")
    parser.add_argument('--lamda',default=0.95,type=float,help="lambda of GAE")
    parser.add_argument('--eps_clip',default=0.2,type=float,help="clip of PPO")
    parser.add_argument('--entropy_coef',default=0.01,type=float,help="entropy coefficient")
    parser.add_argument('--entropy_decay',default=0.99,type=float,help="entropy decay")
    parser.add_argument('--lr_actor',default=0.0003,type=float,help="learning rate")
    parser.add_argument('--lr_critic',default=0.001,type=float,help="learning rate")
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--update_rate',default=2048,type=int)
    parser.add_argument('--hidden_dim',default=128,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda")
    parser.add_argument('--seed',default=2,type=int,help="seed") 
    args = parser.parse_args()
    args = vars(args)#{**vars(args)}
    print("超参数信息")
    print(''.join(['=']*80))#''不加分隔符
    tplt = "{:^20}\t{:^20}\t{:^20}"
    # ^居中，<左对齐，>右对齐
    print(tplt.format("Name", "Value", "Type"))#使用 format 方法将三个字符串 "Name"、"Value" 和 "Type" 分别插入到模板的三个占位符中；打印表头
    for k, v in args.items():
        print(tplt.format(k, v, type(v).__name__))
    print(''.join(['=']*80))#打印分隔线

    if args['seed'] is not None:
        all_seeds(args['seed'])#设置随机种子
    

    #env = gym.make("MountainCarContinuous-v0", render_mode="human", goal_velocity=0.0001)#goal_velocity=0.1表示目标速度；render_mode="human"表示渲染模式为人类可见模式；
    env = gym.make("MountainCarContinuous-v0")#创建环境

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"状态空间维度: {state_dim}, 动作维度: {action_dim}")
    args.update({'state_dim': state_dim, 'action_dim': action_dim})
    agent = PPO_agent(args)
    print("智能体创建成功！")

    train(args, env, agent, state_dim)  # 训练
    #test(args, env, agent, state_dim)  # 测试

if __name__ == '__main__':
    main()