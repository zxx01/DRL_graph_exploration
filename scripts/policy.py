import os
import gc
import random
from collections import deque
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
import Networks
import envs.exploration_env as robot
from PrioritizedReplayBuffer import PrioritizedReplayBuffer

class DeepQ(object):
    def __init__(self, case_path, model_name):
        # define the local file path
        self.case_path = case_path
        self.weights_path = "../data/torch_weights/" + self.case_path
        self.reward_data_path = "../data/reward_data/" + self.case_path
        self.object_path = '../data/training_object_data/' + self.case_path
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        if not os.path.exists(self.reward_data_path):
            os.makedirs(self.reward_data_path)
        if not os.path.exists(self.object_path):
            os.makedirs(self.object_path)
        data_all = pd.DataFrame({"Step": [], "Reward": []})
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)

        # setup parameters for RL
        self.BATCH = 64
        self.REPLAY_MEMORY = 1e5
        self.GAMMA = 0.99
        self.OBSERVE = 5e3
        self.EXPLORE = 1e6
        self.epoch = 1e4
        # self.TARGET_UPDATE = 15000 if model_name == "GCN" else 9000
        
        # 软更新参数
        self.TAU = 0.01
        
        # 探索参数
        self.FINAL_EPSILON = 0
        self.INITIAL_EPSILON = 0.9
        self.max_grad_norm = 0.5

        # setup environment parameters
        self.map_size = 40
        
        # 使用优先经验回放缓冲区
        self.buffer = PrioritizedReplayBuffer(self.REPLAY_MEMORY)
        
        # setup training
        self.step_t = 0
        self.epsilon = self.INITIAL_EPSILON
        self.temp_loss = 0
        self.total_reward = np.empty([0, 0])

    def running(self, model, modelTarget, test=False, double_dqn=True, exploration_method=None):
        data_all = pd.read_csv(self.reward_data_path + "reward_data.csv")
        temp_i = 0
        Test = test
        
        # 如果外部指定了探索方法，则使用指定的方法
        if exploration_method in ["noisy", "epsilon", "bayesian"]:
            method = exploration_method
        else:
            # 根据模型类型自动选择合适的探索策略
            if hasattr(model, 'reset_noise'):  # 检查模型是否有reset_noise方法
                method = "noisy"  # 当使用带噪声的模型时，使用Noisy Networks探索
            else:
                method = "epsilon"  # 默认使用epsilon-greedy
            
        env = robot.ExplorationEnv(self.map_size, 0, Test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = model
        target_net = modelTarget
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-5)

        # 打印训练配置
        print(f"训练配置:")
        print(f"- 探索策略: {method}")
        print(f"- 使用Double DQN: {double_dqn}")
        print(f"- 使用优先经验回放: 是")
        print(f"- 使用软更新: 是 (TAU={self.TAU})")
        print(f"- 学习率: 1e-5")
        print(f"- Batch大小: {self.BATCH}")
        print(f"- 缓冲区大小: {self.REPLAY_MEMORY}")
        print(f"- 折扣因子(GAMMA): {self.GAMMA}")
        
        # 根据模型类型检查是否正确匹配探索策略
        if method == "noisy" and not hasattr(model, 'reset_noise'):
            print(f"警告: 'noisy'策略仅适用于带有reset_noise方法的模型。")
            print(f"自动切换到epsilon-greedy策略。")
            method = "epsilon"
        
        temp_reward_data = []
        temp_loss_data = []
        while temp_i < self.epoch:
            self.step_t += 1
            temp_i += 1

            # 获取输入数据(X, A)
            adjacency, featrues, globals_features, fro_size = env.graph_matrix()
            node_size = adjacency.shape[0]
            key_size = node_size - fro_size
            s_t = self.data_process([adjacency, featrues])
            
            # 获取动作和奖励
            all_actions = env.actions_all_goals()
            rewards = env.rewards_all_goals(all_actions)

            # 根据不同的探索策略选择动作
            if method == "noisy":
                # 重置噪声
                if hasattr(policy_net, 'reset_noise'):
                    policy_net.reset_noise()
                # 使用Noisy Networks选择动作，保持训练模式以启用噪声进行探索
                readout_t = self.test(s_t, 0.0, device, policy_net, keep_train=True)  # 保持训练模式
                readout_t = readout_t.cpu().detach().numpy()
                action_index = np.argmax(readout_t[-fro_size:])
                state = "noisy"
            elif method == "bayesian":
                # 使用Bayesian方法选择动作
                if self.epsilon > self.FINAL_EPSILON and self.step_t > self.OBSERVE:
                    self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

                # 保持训练模式以启用dropout的贝叶斯不确定度
                readout_t = self.test(s_t, self.epsilon, device, policy_net, keep_train=True)  
                readout_t = readout_t.cpu().detach().numpy()
                action_index = np.argmax(readout_t[-fro_size:])
                state = "bayesian"
            else:  # epsilon-greedy
                # e-greedy scale down epsilon
                if self.epsilon > self.FINAL_EPSILON and self.step_t > self.OBSERVE:
                    self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE
                
                # epsilon-greedy策略
                if random.random() <= self.epsilon:
                    # 随机选择动作
                    action_index = random.randint(0, fro_size - 1)
                    state = "random"
                else:
                    # 贪婪选择动作
                    readout_t = self.test(s_t, 0.0, device, policy_net)  # 使用默认的eval模式
                    readout_t = readout_t.cpu().detach().numpy()
                    action_index = np.argmax(readout_t[-fro_size:])
                    state = "greedy"
            
            # 创建动作向量
            a_t = np.zeros([node_size])
            a_t[key_size + action_index] = 1
            
            # 选择对应的动作和奖励
            actions = all_actions[key_size + action_index]
            r_t = rewards[key_size + action_index]
            
            # 执行动作
            for act in actions:
                _, done, _ = env.step(act)
            
            # 终止条件
            current_done = done or env.loop_clo
            
            # 获取下一个状态
            adjacency, featrues, globals_features, fro_size1 = env.graph_matrix()
            s_t1 = self.data_process([adjacency, featrues])
            
            # 计算TD误差用于优先经验回放
            with torch.no_grad():
                if double_dqn:
                    # Double DQN: 策略网络选择动作，目标网络评估
                    # 策略网络在选择动作时应该使用确定性行为
                    next_q_values_policy = self.test(s_t1, 0.0, device, policy_net)
                    next_q_values_policy = next_q_values_policy.cpu().numpy()
                    
                    # 目标网络评估
                    next_q_values_target = self.test(s_t1, 0.0, device, target_net)
                    next_q_values_target = next_q_values_target.cpu().numpy()
                    
                    # 获取前景点的Q值
                    frontier_q_policy = next_q_values_policy[-fro_size1:]
                    frontier_q_target = next_q_values_target[-fro_size1:]
                    
                    # 策略网络选择动作
                    next_best_action = np.argmax(frontier_q_policy)
                    # 目标网络评估动作
                    next_max_q = frontier_q_target[next_best_action]
                else:
                    # 标准DQN:
                    next_q_values_target = self.test(s_t1, 0.0, device, target_net)
                    next_q_values_target = next_q_values_target.cpu().numpy()
                    next_max_q = np.max(next_q_values_target[-fro_size1:])
                
                # 对于随机动作，当前Q值可能不存在
                if method == "epsilon" and state == "random":
                    current_q = 0  # 随机动作没有Q值
                else:
                    current_q = readout_t[key_size + action_index]
                
                td_error = abs(r_t + self.GAMMA * next_max_q * (1 - current_done) - current_q)

            # 保存到优先经验回放缓冲区
            self.buffer.push((s_t, a_t, r_t, s_t1, current_done, fro_size1), td_error)

            # 打印状态信息
            if method == "epsilon" or method == "bayesian":
                print("TIMESTEP", self.step_t, "/ STATE", state, "/ EPSILON", self.epsilon,
                      "/ EXPLORED", env.status(), "/ REWARD", r_t, "/ Terminal", current_done, "\n")
            else:
                print("TIMESTEP", self.step_t, "/ STATE", state, "/ Q_MAX %e" % np.max(readout_t),
                      "/ EXPLORED", env.status(), "/ REWARD", r_t, "/ Terminal", current_done, "\n")

            # 训练步骤
            if self.step_t > self.OBSERVE:
                # 软更新目标网络
                self.soft_update(target_net, policy_net)

                # 从优先经验回放缓冲区采样
                minibatch, indices, weights = self.buffer.sample(self.BATCH)
                weights = torch.FloatTensor(weights).to(device)

                # 获取批量变量
                s_j_batch = [d[0] for d in minibatch] # type
                s_j1_batch = [d[3] for d in minibatch]
                s_j_loader = DataLoader(s_j_batch, batch_size=self.BATCH)
                s_j1_loader = DataLoader(s_j1_batch, batch_size=self.BATCH)
                for batch in s_j_loader:
                    s_j_batch = batch
                for batch1 in s_j1_loader:
                    s_j1_batch = batch1

                r_batch = [d[2] for d in minibatch]
                
                if double_dqn:
                    # Double DQN: 策略网络选择动作，目标网络评估
                    # 策略网络在选择动作时应该使用确定性行为
                    q_values_policy = self.test(s_j1_batch, 0.0, device, policy_net)
                    q_values_policy = q_values_policy.cpu().detach().numpy()
                    
                    # 目标网络评估
                    q_values_target = self.test(s_j1_batch, 0.0, device, target_net)
                    q_values_target = q_values_target.cpu().detach().numpy()
                else:
                    # 标准DQN: 只使用目标网络
                    q_values_target = self.test(s_j1_batch, 0.0, device, target_net)
                    q_values_target = q_values_target.cpu().detach().numpy()
                
                a_batch = np.array([])
                y_batch = np.array([])
                start_p = 0
                for i, _ in enumerate(minibatch):
                    terminal = minibatch[i][4]
                    action_space = minibatch[i][5]
                    act = minibatch[i][1]
                    a_batch = np.append(a_batch, act)
                    node_space = len(act)

                    temp_y = np.zeros(node_space)
                    index = np.argmax(act)
                    if terminal:
                        temp_y[index] = r_batch[i]
                    else:
                        if double_dqn:
                            # Double DQN: 策略网络选择动作，目标网络评估
                            # 获取当前样本的临近点Q值
                            temp_range_policy = q_values_policy[start_p:start_p + node_space]
                            # 只考虑前景点（动作空间）
                            frontier_q_policy = temp_range_policy[-action_space:]
                            # 策略网络选择最佳动作
                            best_action = np.argmax(frontier_q_policy)
                            
                            # 同样获取目标网络的Q值
                            temp_range_target = q_values_target[start_p:start_p + node_space]
                            frontier_q_target = temp_range_target[-action_space:]
                            # 目标网络评估策略网络选择的动作
                            max_q = frontier_q_target[best_action]
                        else:
                            # 标准DQN
                            temp_range = q_values_target[start_p:start_p + node_space]
                            # 只考虑前景点（动作空间）
                            frontier_q = temp_range[-action_space:]
                            # 直接选择最大Q值
                            max_q = np.max(frontier_q)
                        
                        temp_y[index] = r_batch[i] + self.GAMMA * max_q
                    start_p += node_space
                    y_batch = np.append(y_batch, temp_y)

                # print("A_BATCH1", a_batch.shape)
                # 使用重要性权重进行训练
                self.train(s_j_batch, a_batch, y_batch, device, policy_net, optimizer, weights)
                # print("A_BATCH2", a_batch.shape)
                
                # 更新优先级
                with torch.no_grad():
                    # 获取当前Q值时使用确定性模式，以便计算准确的TD误差
                    current_q_values = self.test(s_j_batch, 0.0, device, policy_net)
                    current_q_values = current_q_values.cpu().numpy()
                    
                    # print("CURRENT_Q", current_q_values.shape)
                    
                    if double_dqn:
                        # 策略网络在TD目标计算时使用确定性模式
                        q_values_policy = self.test(s_j1_batch, 0.0, device, policy_net)
                        q_values_policy = q_values_policy.cpu().detach().numpy()
                        
                        # 目标网络评估
                        q_values_target = self.test(s_j1_batch, 0.0, device, target_net)
                        q_values_target = q_values_target.cpu().detach().numpy()
                    else:
                        next_q_values = self.test(s_j1_batch, 0.0, device, target_net)
                        next_q_values = next_q_values.cpu().numpy()
                    
                    td_errors = []
                    start_p = 0  # 跟踪在next_q_values中的起始位置
                    for i, (act, r, done) in enumerate(zip([d[1] for d in minibatch], r_batch, [d[4] for d in minibatch])):
                        node_space = len(act)  # 当前样本的节点数量
                        action_space = minibatch[i][5]  # 当前样本的动作空间大小
                        
                        # 正确获取当前状态下选择的动作的Q值
                        action_idx = np.argmax(act)  # 找到动作的索引
                        q_idx = start_p + action_idx  # 获取在current_q_values中的索引位置
                        current_q = current_q_values[q_idx]  # 获取Q值
                        
                        if not done:
                            if double_dqn:
                                # 策略网络选择动作
                                temp_range_policy = q_values_policy[start_p:start_p + node_space]
                                temp_range_policy = temp_range_policy[-action_space:]
                                best_action = np.argmax(temp_range_policy)
                                
                                # 目标网络评估动作
                                temp_range_target = q_values_target[start_p:start_p + node_space]
                                temp_range_target = temp_range_target[-action_space:]
                                next_max_q = temp_range_target[best_action]
                            else:
                                temp_range = next_q_values[start_p:start_p + node_space]
                                temp_range = temp_range[-action_space:]
                                next_max_q = np.max(temp_range)
                            
                            td_error = abs(r + self.GAMMA * next_max_q - current_q)
                        else:
                            td_error = abs(r - current_q)
                        
                        td_errors.append(td_error)
                        start_p += node_space  # 更新下一个样本的起始位置
                    
                    self.buffer.update_priorities(indices, td_errors)

                temp_loss_data.append([self.step_t, self.temp_loss])

            if done:
                del env
                gc.collect()
                env = robot.ExplorationEnv(self.map_size, 0, Test)
                done = False

            new_row_df = pd.DataFrame([{"Step": self.step_t, "Reward": r_t}])
            data_all = pd.concat([data_all, new_row_df], ignore_index=True)
            self.total_reward = np.append(self.total_reward, r_t)

            if self.step_t % 5e4 == 0:
                torch.save(policy_net.state_dict(), self.weights_path + 'MyModel.pt')
            if self.step_t > 1000:
                new_average_reward = np.average(self.total_reward[len(self.total_reward) - 1000:])
                if self.step_t % 1e2 == 0:
                    temp_reward_data.append([self.step_t, new_average_reward])

        np.savetxt(self.object_path + "temp_reward.csv", temp_reward_data, delimiter=",")
        np.savetxt(self.object_path + "temp_loss.csv", temp_loss_data, delimiter=",")
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)
        torch.save(policy_net.state_dict(), self.object_path + 'Model_Policy.pt')
        torch.save(target_net.state_dict(), self.object_path + 'Model_Target.pt')

    def soft_update(self, target_net, policy_net):
        """软更新目标网络"""
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                self.TAU * policy_param.data + (1.0 - self.TAU) * target_param.data
            )

    def data_process(self, data):
        s_a, s_x = data  # s_a是邻接矩阵，s_x是节点特征
        edge_index = []
        edge_attr = []
        edge_set = set()
        
        # 将邻接矩阵转换为COO格式的边索引和边属性
        for a_i in range(np.shape(s_a)[0]):
            for a_j in range(np.shape(s_a)[1]):
                if (a_i, a_j) in edge_set or (a_j, a_i) in edge_set \
                        or s_a[a_i][a_j] == 0:
                    continue
                # 添加边
                edge_index.append([a_i, a_j])
                edge_attr.append(s_a[a_i][a_j])
                # 添加反向边（无向图）
                if a_i != a_j:
                    edge_index.append([a_j, a_i])
                    edge_attr.append(s_a[a_j][a_i])
                edge_set.add((a_i, a_j))
                edge_set.add((a_j, a_i))
        
        # 转换为PyTorch张量
        edge_index = torch.tensor(np.transpose(edge_index), dtype=torch.long)
        x = torch.tensor(s_x, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # 创建PyTorch Geometric的Data对象
        state = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return state

    def cost(self, pred, target, action):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        readout_action = torch.mul(pred_flat, action)
        loss = torch.pow(readout_action - target_flat, 2).sum() / self.BATCH
        return loss

    def train(self, data, action, y, device, model, optimizer, weights=None):
        model.train()
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data, 0.5, batch=data.batch)
        y = torch.tensor(y).to(device)
        action = torch.tensor(action).to(device)
        
        # 使用重要性权重计算损失
        if weights is not None:
            loss = (weights * self.cost(out, y, action)).mean()
        else:
            loss = self.cost(out, y, action)
            
        self.temp_loss = loss.item()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
        optimizer.step()

    def test(self, data, prob, device, model, target_model=None, double_dqn=False, keep_train=False):
        """
        使用模型预测Q值
        
        Args:
            data: 输入数据
            prob: 使用的概率参数
            device: 设备
            model: 主模型
            target_model: 目标模型，用于Double DQN
            double_dqn: 是否使用Double DQN
            keep_train: 是否保持模型的训练模式（对于NoisyNet很重要）
            
        Returns:
            Q值预测
        """
        # 只有在不需要保持训练模式时才切换到eval模式
        if not keep_train:
            model.eval()
            
        data = data.to(device)
        
        if double_dqn and target_model is not None:
            # Double DQN: 使用策略网络选择动作，使用目标网络评估动作
            target_model.eval()  # 目标网络始终使用eval模式
            with torch.no_grad():
                return target_model(data, prob)
        else:
            # 标准DQN或只需要策略网络的预测
            with torch.no_grad():
                return model(data, prob)


class A2C(object):
    def __init__(self, case_path):
        # define the local file path
        self.case_path = case_path
        self.weights_path = "../data/torch_weights/" + self.case_path
        self.reward_data_path = "../data/reward_data/" + self.case_path
        self.object_path = '../data/training_object_data/' + self.case_path
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        if not os.path.exists(self.reward_data_path):
            os.makedirs(self.reward_data_path)
        if not os.path.exists(self.object_path):
            os.makedirs(self.object_path)
        data_all = pd.DataFrame({"Step": [], "Reward": []})
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)

        # setup parameters for RL
        self.GAMMA = 0.99
        self.EXPLORE = 1e6  # 5e5
        self.epoch = 1e4  # 1e4
        self.nstep = 40
        self.ent_coef = 0.01
        self.vf_coef = 0.25
        self.max_grad_norm = 0.5

        # setup memory
        self.buffer = deque()
        # setup environment parameters
        self.map_size = 40
        # setup training
        self.step_t = 0
        self.temp_loss = 0
        self.entro = 0
        self.total_reward = np.empty([0, 0])

    def running(self, actor, critic, test=False):
        data_all = pd.read_csv(self.reward_data_path + "reward_data.csv")
        temp_i = 0
        Test = test
        env = robot.ExplorationEnv(self.map_size, 0, Test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = actor
        value_net = critic
        params = list(policy_net.parameters()) + list(value_net.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-5)

        temp_reward_data = []
        temp_loss_data = []
        while temp_i < self.epoch:
            self.step_t += 1
            temp_i += 1
            # get the input data (X, A)
            adjacency, featrues, globals_features, fro_size = env.graph_matrix()
            node_size = adjacency.shape[0]
            key_size = node_size - fro_size
            s_t, b_t = self.data_process([adjacency, featrues], device)
            mask = np.zeros([node_size])
            mask[-fro_size:] = 1

            # get the output reward (Y)
            all_actions = env.actions_all_goals()
            rewards = env.rewards_all_goals(all_actions)

            # choose an action
            readout_t = self.test(s_t, b_t, mask, device,
                                  policy_net).view(-1).cpu().detach().numpy()
            val = self.test(s_t, b_t, mask, device, value_net).item()

            action_index = np.random.choice(fro_size, 1, p=readout_t)[0]
            action_index = key_size + action_index

            a_t = np.zeros([node_size])
            a_t[action_index] = 1

            # choose an action
            actions = all_actions[action_index]

            # get reward
            r_t = rewards[action_index]

            # move to the next view point
            for act in actions:
                _, done, _ = env.step(act)

            # terminal for RL value calculation
            current_done = done or env.loop_clo

            # get next state
            adjacency, featrues, globals_features, fro_size1 = env.graph_matrix()
            s_t1, b_t1 = self.data_process([adjacency, featrues], device)
            mask = np.zeros([adjacency.shape[0]])
            mask[-fro_size1:] = 1

            last_value = self.test(s_t1, b_t1, mask, device, value_net).item()

            # save to buffer
            self.buffer.append(
                (s_t, a_t, r_t, s_t1, current_done, fro_size, val))

            # training step
            if len(self.buffer) == self.nstep:
                # get the batch variables
                s_j_batch = [d[0] for d in self.buffer]
                s_j1_batch = [d[3] for d in self.buffer]
                s_j_loader = DataLoader(s_j_batch, batch_size=self.nstep)
                for batch in s_j_loader:
                    s_j_batch = batch
                r_batch = [d[2] for d in self.buffer]
                value_j = [d[6] for d in self.buffer]

                discount_rewards = []
                ret = last_value
                for i in reversed(range(len(self.buffer))):
                    terminal = self.buffer[i][4]
                    ret = r_batch[i] + self.GAMMA * ret * (1.0-terminal)
                    discount_rewards.append(ret)
                discount_rewards = discount_rewards[::-1]

                a_batch = np.array([])
                y_adv_batch = np.array([])
                mask_batch = np.array([])
                for i, _ in enumerate(self.buffer):
                    # for i in range(0, len(self.buffer)):
                    action_space = self.buffer[i][5]
                    act = self.buffer[i][1]
                    a_batch = np.append(a_batch, act)
                    node_space = len(act)
                    temp_mask = np.zeros(node_space)
                    temp_mask[-action_space:] = 1
                    temp_y = np.zeros(node_space)
                    index = np.argmax(act)
                    # get policy loss
                    temp_y[index] = discount_rewards[i] - value_j[i]
                    y_adv_batch = np.append(y_adv_batch, temp_y)
                    mask_batch = np.append(mask_batch, temp_mask)

                # perform gradient step
                self.train(s_j_batch, a_batch, mask_batch, discount_rewards, y_adv_batch,
                           device, policy_net, value_net, optimizer)
                temp_loss_data.append([self.step_t, self.temp_loss])
                self.buffer.clear()

            print("TIMESTEP", self.step_t,
                  "/ Loss", self.temp_loss, "/ Entropy", self.entro,
                  "/ EXPLORED", env.status(), "/ REWARD", r_t, "/ Terminal", current_done, "\n")

            if done:
                del env
                gc.collect()
                env = robot.ExplorationEnv(self.map_size, 0, Test)
                done = False

            new_row_df = pd.DataFrame([{"Step": self.step_t, "Reward": r_t}])
            data_all = pd.concat([data_all, new_row_df], ignore_index=True)
            self.total_reward = np.append(self.total_reward, r_t)

            # save progress every 50000 iterations
            if self.step_t % 5e4 == 0:
                torch.save(policy_net.state_dict(),
                           self.weights_path + 'MyModel.pt')
            if self.step_t > 1000:
                new_average_reward = np.average(
                    self.total_reward[len(self.total_reward) - 1000:])
                if self.step_t % 1e2 == 0:
                    temp_reward_data.append([self.step_t, new_average_reward])

        np.savetxt(self.object_path + "temp_reward.csv",
                   temp_reward_data, delimiter=",")
        np.savetxt(self.object_path + "temp_loss.csv",
                   temp_loss_data, delimiter=",")
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)
        torch.save(policy_net.state_dict(),
                   self.object_path + 'Model_Policy.pt')
        torch.save(value_net.state_dict(), self.object_path + 'Model_Value.pt')

    def data_process(self, data, device):
        s_a, s_x = data
        edge_index = []
        edge_attr = []
        edge_set = set()
        for a_i in range(np.shape(s_a)[0]):
            for a_j in range(np.shape(s_a)[1]):
                if (a_i, a_j) in edge_set or (a_j, a_i) in edge_set \
                        or s_a[a_i][a_j] == 0:
                    continue
                edge_index.append([a_i, a_j])
                edge_attr.append(s_a[a_i][a_j])
                if a_i != a_j:
                    edge_index.append([a_j, a_i])
                    edge_attr.append(s_a[a_j][a_i])
                edge_set.add((a_i, a_j))
                edge_set.add((a_j, a_i))
        edge_index = torch.tensor(np.transpose(edge_index), dtype=torch.long)
        x = torch.tensor(s_x, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        state = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        batch = torch.zeros(np.shape(s_a)[0], dtype=int).to(device)
        return state, batch

    def policy_cost(self, prob, advantages, action, mask):
        prob_flat = prob.view(-1)
        advantages_flat = advantages.view(-1)
        advantages_flat = torch.masked_select(
            advantages_flat, mask).to(torch.float32)
        action = torch.masked_select(action, mask).to(torch.float32)
        log_prob = prob_flat.log().to(torch.float32)
        policy_loss = -torch.mul(log_prob, advantages_flat)
        policy_loss = torch.mul(policy_loss, action).sum() / self.nstep
        return policy_loss

    def value_cost(self, pred, target):
        pred_flat = pred.view(-1).to(torch.float32)
        target_flat = target.view(-1).to(torch.float32)
        loss = F.mse_loss(pred_flat, target_flat)
        return loss

    def entropy_loss(self, prob):
        prob_flat = prob.view(-1).detach().to(torch.float32)
        entro = -torch.mul(prob_flat.log(), prob_flat).sum() / self.nstep
        self.entro = entro.item()
        return entro

    def train(self, data, action, mask, dis_reward, y_adv,
              device, modelA, modelC, optimizer):
        modelA.train()
        modelC.train()
        data = data.to(device)
        mask = torch.tensor(mask, dtype=bool).to(device)
        optimizer.zero_grad()
        actor_out = modelA(data, mask, batch=data.batch)
        critic_out = modelC(data, mask, batch=data.batch)
        eps = 1e-35
        actor_out = actor_out + eps
        y_adv = torch.tensor(y_adv).to(device)
        dis_reward = torch.tensor(dis_reward).to(device)
        action = torch.tensor(action).to(device)
        actor_loss = self.policy_cost(actor_out, y_adv, action, mask)
        critic_loss = self.value_cost(critic_out, dis_reward)
        entropy_loss = self.entropy_loss(actor_out)
        loss = actor_loss - entropy_loss * self.ent_coef + critic_loss * self.vf_coef
        self.temp_loss = loss.item()
        loss.backward()
        params = list(modelA.parameters()) + list(modelC.parameters())
        for param in params:
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
        optimizer.step()

    def test(self, data, batch, mask, device, model):
        model.eval()
        data = data.to(device)
        mask = torch.tensor(mask, dtype=bool).to(device)
        pred = model(data, mask, batch)
        return pred


if __name__ == "__main__":
    case_path = "test_case"
    training = A2C(case_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modela = Networks.PolicyGCN()
    modelc = Networks.ValueGCN()
    modela.to(device)
    modelc.to(device)
    training.running(modela, modelc)
