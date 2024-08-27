import math
import random
import numpy as np

epsilon = 0.4
num_sub_decision_makers = 1
num_arms = 15
reward_value = 1
para = 10


def compute_similarity(list1, list2):
    count = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            count += 1
    similarity = count / len(list1)
    return similarity


class SubDecisionMaker:
    def __init__(self, alpha_list, beta_list):
        self.alpha_list = alpha_list
        self.beta_list = beta_list
        self.decision_list = []
        self.percent = 1 / num_sub_decision_makers

    def thompson_sampling(self):
        # 使用Thompson采样生成决策结果
        # 这里假设生成的决策结果为0到15的整数
        probabilities = [np.random.beta(self.alpha_list[i], self.beta_list[i]) for i in range(len(self.alpha_list))]
        decision = np.argmax(probabilities)
        return decision

    def reset(self):
        self.decision_list = []

    def record_decision(self, decision):
        self.decision_list.append(decision)

    def get_decision(self):
        return self.decision_list

    def get_percent(self):
        return self.percent

    def update_sub_decision_maker(self, best_arms, reward, index, decisions):
        for i, decision in enumerate(decisions):
            if decision == best_arms[i] and reward == 1:
                self.alpha_list[decision] = self.alpha_list[decision] + reward_value / index
            if decision == best_arms[i] and reward == 0:
                self.beta_list[decision] = self.beta_list[decision] + reward_value / para / index

    def compare(self, best_arms, before, after):
        similarity = compute_similarity(self.decision_list[before:after], best_arms)
        return similarity

    def update_percent(self, new_percent):
        self.percent = new_percent


def initial():
    # 初始化子决策器列表
    # 保守的决策器
    sub_decision_makers = []
    for _ in range(num_sub_decision_makers):
        alpha_list = [1 for _ in range(num_arms)]  # 生成alpha列表
        beta_list = [1 for _ in range(num_arms)]  # 生成beta列表
        sub_decision_maker = SubDecisionMaker(alpha_list, beta_list)
        sub_decision_makers.append(sub_decision_maker)

    return sub_decision_makers


def make_decision(sub_decision_makers):
    arm_counts = [0] * num_arms
    # 统计各个决策结果的计数
    for sub_decision_maker in sub_decision_makers:
        decision = sub_decision_maker.thompson_sampling()
        sub_decision_maker.record_decision(decision)
        arm_counts[decision] += sub_decision_maker.get_percent()

    # 找到计数最大的决策结果
    max_count = max(arm_counts)
    best_arms = [i for i, count in enumerate(arm_counts) if count == max_count]

    # 返回最佳决策结果
    return random.choice(best_arms), sub_decision_makers
