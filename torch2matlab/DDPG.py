
import random
import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, cat
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def shape_check(array, shape):
    assert array.shape == shape, \
        'shape error | array.shape ' + str(array.shape) + ' shape: ' + str(shape)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.Flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_dim),
            nn.Tanh()
        )              

    def forward(self, x):
        x = self.Flatten(x)
        x = self.fc(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1_state = nn.Linear(3, 32)
        self.relu1_state = nn.ReLU()
        
        self.fc1_action = nn.Linear(1, 32)
        self.relu1_action = nn.ReLU()

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 1)

    def forward(self, X):
        x = self.fc1_state(X[0])
        x = self.relu1_state(x)
        a = self.fc1_action(X[1])
        a = self.relu1_action(a)
        h = cat((x, a),1)
        h = self.fc2(h)
        h = self.relu2(h)
        h = self.fc3(h)
        h = self.relu3(h)
        h = self.fc4(h)
        
        return h

class Agent:
    def __init__(self, state_dim, action_dim, action_bound):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.render = False

        # env parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # hyperparameters
        self.GAMMA = 0.95
        self.Actor_learning_rate = 0.0001
        self.Critic_learning_rate = 0.001
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.TAU = 0.001
        self.train_start_limit = 1000

        # ou noise parameters
        self.noise_rho = 0.15
        self.noise_mu = 0
        self.noise_sigma = 0.2
        self.noise_increment = 0.1 # increment: 시간 증분

        # buffer
        self.buffer = deque(maxlen=self.BUFFER_SIZE)

        # model
        self.Actor_model = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device)
        self.Actor_target_model = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device)
        self.Critic_model = Critic(self.state_dim, self.action_dim).to(self.device)
        self.Critic_target_model = Critic(self.state_dim, self.action_dim).to(self.device)


        # model optimizer
        self.Actor_optimizer =  torch.optim.Adam(self.Actor_model.parameters(), lr=self.Actor_learning_rate)
        self.Critic_optimizer = torch.optim.Adam(self.Critic_model.parameters(), lr=self.Critic_learning_rate)

        print(
            'ENV INFO | ',
            'state_dim: ', self.state_dim,
            'action_dim: ', self.action_dim,
            'action_bound: ', self.action_bound
        )

    def sample_append(self, state, action, reward, next_state, done):
        self.buffer.append(
            [
                state,
                action,
                reward,
                next_state,
                done
            ]
        )

    def TD_target(self, reward_list, next_state_list, done_list):
        next_action_list = np.array(next_state_list, dtype=np.float32)
        next_state_list = np.array(next_state_list, dtype=np.float32)

        next_action_list = self.Actor_target_model(
            torch.as_tensor(next_state_list, device=self.device)
        )
        next_Q = self.Critic_target_model(
            [
                torch.as_tensor(next_state_list, device=self.device),
                torch.as_tensor(next_action_list, device=self.device)
                
                # next_action_list,
            ]
        )

        reward_list = np.reshape(reward_list, [self.BATCH_SIZE, 1])
        done_list = np.reshape(done_list, [self.BATCH_SIZE, 1])

        next_Q = next_Q.detach().cpu().numpy()
        # target = reward_list + (1 - done_list) * self.GAMMA * next_Q
        
        target = np.asarray(next_Q)
        for i in range(len(next_Q)):
            if done_list[i]:
                target[i] = reward_list[i]
            else: 
                target[i] = reward_list[i] + self.GAMMA * next_Q[i]

        shape_check(next_Q, (self.BATCH_SIZE, 1))
        shape_check(target, (self.BATCH_SIZE, 1))

        return target

    def update_target_networks(self, model, target_model, TAU):
        with torch.no_grad():
            # for i in range(len(model)):
            #     target_model[i].weight = TAU * model[i].weight + (1-TAU) * target_model[i].weight
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(TAU*param.data + target_param.data*(1.0 - TAU))

    def ou_noise(self, pre_noise):
        nt = np.random.normal(size=self.action_dim)
        noise = pre_noise + self.noise_rho * (self.noise_mu - pre_noise) * self.noise_increment \
             + np.sqrt(self.noise_increment) * self.noise_sigma * nt
        
        return noise

    def get_action(self, state, noise):
        state = np.array(state, dtype=np.float32)
        # print(state)
        action = self.Actor_model(
            torch.as_tensor(
                state, device=self.device
            )
            # torch.tensor(state, device=self.device)
        )
        action = action.detach().cpu().numpy()

        return action + noise

    def actor_train(self, state_list):
        state_list = np.array(state_list, dtype=np.float32)

        self.Actor_model.train()
        action_list = self.Actor_model(
            torch.as_tensor(
                state_list, device=self.device
            )
        )
        # with torch.no_grad():
        Q = self.Critic_model(
            [
                torch.as_tensor(
                    state_list, device=self.device
                ),
                action_list
            ]
        )
        loss = -torch.mean(Q)
        shape_check(action_list, (self.BATCH_SIZE, 1))
        shape_check(Q, (self.BATCH_SIZE, 1))
        # grads = tape.gradient(loss, model_params)
        self.Actor_optimizer.zero_grad()
        loss.backward()
        self.Actor_optimizer.step()
        # self.Actor_optimizer.apply_gradients(zip(grads, model_params))

    def critic_train(self, target_list, state_list, action_list):
        # model_params = self.Critic_model.trainable_variables
        state_list = np.array(state_list, dtype=np.float32)
        action_list = np.array(action_list, dtype=np.float32)
        target_list = torch.tensor(target_list, device=self.device)

        self.Critic_model.train()
        predict_Q = self.Critic_model(
            [
            torch.as_tensor(
                state_list, device=self.device
            ),            
            torch.as_tensor(
                action_list, device=self.device
            )
            ]
        )
        # loss = tf.reduce_mean(tf.square(target_list - predict_Q))
        loss = torch.mean(torch.square(target_list - predict_Q))
        shape_check(predict_Q, (self.BATCH_SIZE, 1))
        shape_check(target_list, (self.BATCH_SIZE, 1))
        # grads = tape.gradient(loss, model_params)
        # self.Critic_optimizer.apply_gradients(zip(grads, model_params))
        self.Critic_optimizer.zero_grad()
        loss.backward()
        self.Critic_optimizer.step()

    def train(self):
        
        batch = random.sample(self.buffer, self.BATCH_SIZE)

        state_list = [sample[0][0] for sample in batch]
        action_list = [sample[1][0] for sample in batch]
        reward_list = [sample[2][0] for sample in batch]
        next_state_list = [sample[3][0] for sample in batch]
        done_list = [sample[4][0] for sample in batch]

        target_list = self.TD_target(reward_list, next_state_list, done_list)

        self.critic_train(target_list, state_list, action_list)
        self.actor_train(state_list)
        
        self.update_target_networks(
            self.Actor_model, 
            self.Actor_target_model, 
            self.TAU
        )
        self.update_target_networks(
            self.Critic_model, 
            self.Critic_target_model, 
            self.TAU
        )

    def saveasONNX(self):
        self.Actor_model.eval()
        self.Critic_model.eval()

        dummy_state = torch.randn(1, 3, device=self.device, requires_grad=True)
        dummy_action = torch.randn(1, 1, device=self.device, requires_grad=True)        
        torch.onnx.export(
            self.Actor_model, 
            dummy_state, 
            "DDPG_actor.onnx", 
            verbose=False,
            input_names= ['input']
            )
        torch.onnx.export(
            self.Critic_model, 
            [dummy_state, dummy_action], 
            "DDPG_critic.onnx", 
            verbose=False
            )         

if __name__ == '__main__':

    # env, agent setting
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    agent = Agent(state_dim, action_dim, action_dim)

    # parameters
    score_avg_list = []
    score_avg = 0
    EPISODE = int(1e3)

    # EPISODE start
    for e in range(EPISODE):
        # first step
        state = env.reset()
        # state = state[0] # ...?
        state = np.reshape(state, [1, state_dim])

        # parameters initialize
        score, step, done = 0, 0, 0
        noise = np.zeros(action_dim)
        agent.update_target_networks(
            agent.Actor_model, 
            agent.Actor_target_model, 
            1
        )
        agent.update_target_networks(
            agent.Critic_model, 
            agent.Critic_target_model, 
            1
        )

        while not  done:
            step += 1

            if agent.render == True:
                env.render()

            # action, step
            noise = agent.ou_noise(noise)
            action = agent.get_action(state, noise)
            action = np.clip(action, -action_bound, action_bound)
            next_state, reward, done, _ = env.step(action)

            # reward normalize
            norm_reward = (reward + 8.0) / 8

            # reshape dim
            action = np.reshape(action, [1, action_dim])
            next_state = np.reshape(next_state, [1, state_dim])
            norm_reward =  np.reshape(norm_reward, [1, 1])
            done = np.reshape(done, [1, 1])

            # sampling
            agent.sample_append(
                state,
                action,
                norm_reward,
                next_state,
                done
            )

            # train
            if len(agent.buffer) > agent.train_start_limit:
                agent.train()

            # for next_state
            score += reward[0]
            state = next_state

        # score log save
        score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
        score_avg_list.append(score_avg)

        print(
            'EPISODE: ', e+1,
            'SCORE: ', round(score, 3),
            'SCORE_AVG: ', round(score_avg, 3)
        )

        # model weights save, plotting
        if (e+1) % 10 == 0:
            # agent.Actor_model.save_weights('DDPG', save_format='tf')
            plt.plot(score_avg_list)
            plt.savefig('./DDPG.png')
            # plt.savefig('./DDPG'+str(e)+'e.png')

    agent.saveasONNX()
