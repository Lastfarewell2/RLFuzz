from itertools import count
import sys
import scipy.optimize

from models import *
from replay_memory import Memory
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from running_state import ZFilter
import sys
import numpy as np
import mmap
import os
import time
import re

torch.set_default_tensor_type('torch.DoubleTensor')

gamma = 0.995  # discount factor (default: 0.995)
tau = 0.97  # gae (default: 0.97)
l2_reg = 1e-3  # l2 regularization regression (default: 1e-3)
max_kl = 1e-2  # max kl value (default: 1e-2)
damping = 1e-1  # damping (default: 1e-1)
batch_size = 15000  # random seed (default: 1)
log_interval = 10  # interval between training status logs (default: 10)

num_inputs = 4096
num_actions = 13 * 4096

epsilon = 0.5

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def select_result(state):
    output = list()
    middle = np.where(state > epsilon)[0]
    for item in middle:
        if item % 1000 < ori_size:
            output.append(item)
    return np.array(output)


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return torch.sigmoid(action)


def update_params(batch):
    print(batch.reward)
    print(batch.state)
    print(batch.action)

    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))
    values = value_net(Variable(states))

    returns = torch.zeros(actions.size(0), 1)
    deltas = torch.zeros(actions.size(0), 1)
    advantages = torch.zeros(actions.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).double().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))

        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, max_kl, damping)

reward_sum = 0

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

percent = 0.8

round_num = 0

for i_episode in count(1):
    memory = Memory()

    with open("TRPO_shared_memory", "r+b") as file:
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
        lock = bytearray(mm[:1])
            
        while lock[0] != 3:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
            lock = bytearray(mm[:1])
            time.sleep(0.001)
        
        input_sequence = mm[1:].decode()
        input_sequence = input_sequence[:input_sequence.find("\x00")]
            
        observation = [int(input_sequence[i:i+2], 16) for i in range(0, len(input_sequence), 2)]
        ori_size = len(observation)

        if len(observation) < 4096:
            observation.extend([0 for _ in range(4096 - len(observation))])
        else:
            observation = observation[:4096]

        observation = running_state(np.array(observation))

        action_score = select_action(observation)
        action_score = action_score.data[0].numpy()
        
        action_save = action_score.copy()
        
        print(percent)
        
        actions = action_score.argsort()[- int(percent * action_score.size) : ]
        
        actions.sort()
        
        percent -= 0.4 / 100

        if round_num > 100:
            percent = 0.4
            
        print(percent)
        
        print(actions.size)
            
        result_string = ''
            
        for action in actions:
            be_re = int(action // 4096)
            af_re = int(action % 4096)
            result_string += (format(be_re, 'x').zfill(3) + format(af_re, 'x').zfill(3))
                
        result_string += "00e001\0"
            
        # RL take action and get next observation and reward
        # 将文件映射为内存
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
        
        # 获取互斥锁
        lock = bytearray(mm[:1])
            
        mm[1:] = b'\x00' * (len(mm) - 1)
        
        mm[1: len(result_string) + 1] = result_string.encode()

        # 释放互斥锁
        lock[0] = 0
    
        # 将修改后的互斥锁状态写回共享内存文件中
        mm.seek(0)
        mm.write(lock)
        
        while lock[0] != 4:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
            lock = bytearray(mm[:1])
            time.sleep(0.001)
                
        output_sequence = mm[1:].decode()
        output_sequence = output_sequence[:output_sequence.find("\x00")]
            
        ob_save = output_sequence.split(',')[:-1];
            
        print(len(ob_save))
            
        lock = bytearray(mm[:1])
        lock[0] = 1
            
        mm.seek(0)
        mm.write(lock)
            
        while lock[0] != 5:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
            lock = bytearray(mm[:1])
            time.sleep(0.001)
                
        output_sequence = mm[1:].decode()
        output_sequence = output_sequence[:output_sequence.find("\x00")]
            
        rewards_save = output_sequence.split(',')[:-1];
            
        print(len(rewards_save))
            
        lock = bytearray(mm[:1])
        lock[0] = 1
            
        mm.seek(0)
        mm.write(lock)
            
        rewards = np.array([int(element) for element in rewards_save])
            
        while lock[0] != 6:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
            lock = bytearray(mm[:1])
            time.sleep(0.001)
                
        output_sequence = mm[1:].decode()
        output_sequence = output_sequence[:output_sequence.find("\x00")]
            
        actions_save = output_sequence.split(',')[:-1];
        actions = np.array([int(element) for element in actions_save])
            
        print(len(actions_save))
            
        mm.write(b"\x01" + b"\x00" * (10000000 - 1))
    	
        # 解除内存映射
        mm.close()
        
    done = 0

    reward_sum += sum(rewards)

    mask = 1
    if done:
        mask = 0
        
    count = 0
    
    rewards_get = np.zeros(num_actions)
    
    total_rewards = 0;
    
    for action in actions:
        total_rewards += rewards[count] * action_save[action]  
        count += 1;
     
    print(total_rewards)

    if done:
        break
        
    memory.push(observation, np.array(action_save), mask, np.array([]), total_rewards)

    batch = memory.sample()
    update_params(batch)
    
    round_num += 1
    
