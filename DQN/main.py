from RL_brain import DeepQNetwork
import sys
import numpy as np
import mmap
import os
import time
import re

max_n = 4096
n_actions = 13 * 4096

def run_maze():
    output_percent = 0.8
    round_num = 0

    while True:
        with open("DQN_shared_memory", "r+b") as file:
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

            observation = np.array(observation)

            # RL choose action based on observation
            actions = RL.choose_action(observation, output_percent, ori_size)
            actions.sort()
            
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

        output_percent -= 0.4 / 100

        if round_num > 100:
            output_percent = 0.4

        count = 0
        # !! restore transition
        for action in actions:
        
            if count >= len(ob_save):
                break
                
            sequence = ob_save[count]
            observation_ = [int(sequence[i:i+2], 16) for i in range(0, len(sequence), 2)]
            ori_size = len(observation_)

            if len(observation_) < 4096:
                observation_.extend([0 for _ in range(4096 - len(observation_))])
            else:
                observation_ = observation_[:4096]
                
            observation_ = np.array(observation_)
            
            if rewards[count] == 1 or round_num == 0:
                RL.store_transition(observation, actions[count], rewards[count], observation_)
            
            count += 1

        n_transition = RL.get_transition()

        if n_transition > 200:
            RL.learn()

        round_num += 1


if __name__ == "__main__":
    RL = DeepQNetwork(n_actions, max_n,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=5,
                      memory_size=2000)
    run_maze()
    RL.plot_cost()
