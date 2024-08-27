import numpy as np
import sys
import json
import Dicision_Maker
import mmap
import os
import time
import re

N = 15
epsilon = 0.2  # 贪婪系数

def pull(n, Epsilon, SubDecisionMakers):

    # 通过一致分布的随机数来确定是搜索还是利用
    exploration_flag = True if np.random.uniform() <= Epsilon else False

    # 如果选择探索
    if exploration_flag:
        decision = int(min(n - 1, np.floor(n * np.random.uniform())))
        for SubDecisionMaker in SubDecisionMakers:
            SubDecisionMaker.record_decision(decision)

    # 如果选择利用
    else:
        decision, SubDecisionMakers = Dicision_Maker.make_decision(SubDecisionMakers)

    return decision, SubDecisionMakers
    
    
def renew_param(sub_decision_makers, actions, reward, sequence, before, after):

    rank = list()
    for sub_decision_maker in sub_decision_makers:
        similarity = sub_decision_maker.compare(actions, before, after)
        rank.append(similarity)

    total_percent = sum(rank)
    count = 0
    
    for sub_decision_maker in sub_decision_makers:
        
        sub_decision_maker.update_sub_decision_maker(actions, reward, sequence, sub_decision_maker.get_decision()[before:after])
        if reward == 1:
            sub_decision_maker.update_percent(rank[count] / total_percent)
        
        count += 1


rewards = 0
done = 1

while True:
    
    with open("ts_shared_memory", "r+b") as file:
    
        if done == 1:
            sub_decision_makers = Dicision_Maker.initial()
            
            print("yes")
            
            done = 0
            
        actions = []
        
        for i in range(1000 * 50):
            action, sub_decision_makers = pull(N, epsilon, sub_decision_makers)
            actions.append(action)
        
        result_string = ''.join([str(action).zfill(2) for action in actions])
        
        print(result_string[:1000])
        
        result_string += '\0'
        
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

        while lock[0] != 2:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
            lock = bytearray(mm[:1])
            time.sleep(0.001)
    	
        reward = mm[1:].decode()
        reward = reward[:reward.find("\x00")]
        
        lock = bytearray(mm[:1])
        lock[0] = 1
            
        mm.seek(0)
        mm.write(lock)
        
        while lock[0] != 3:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
            lock = bytearray(mm[:1])
            time.sleep(0.001)
    	
        sequence_num = mm[1:].decode()
        sequence_num = sequence_num[:sequence_num.find("\x00")]
    	
        mm.write(b"\x01" + b"\x00" * (10000000 - 1))
    	
        # 解除内存映射
        mm.close()
    	
    sequences, done = sequence_num.split(',')
    
    sequences = sequences.split(';')[:-1]
    
    done = int(done)
    
    before = 0
    count = 0
    for sequence in sequences:
        sequence_n = int(sequence)
        renew_param(sub_decision_makers, actions[before:before + sequence_n], int(reward[count]), sequence_n, before, before + sequence_n)
        before += sequence_n
        count += 1
        if count == 1000:
            before = 0
            
    for sub_decision_maker in sub_decision_makers:
        sub_decision_maker.reset()
    
    print("round ends")

