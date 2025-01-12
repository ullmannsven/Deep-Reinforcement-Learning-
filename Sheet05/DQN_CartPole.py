import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def deep_q_learning(env=gym.make('CartPole-v1'), episodes=20000, max_steps=200, gamma=0.99, epsilon=1.0, memory_size=100000, batch_size=32, network_sync_rate=50000, learning_rate=0.001):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_network = QNetwork(state_size, action_size)
    target_network = QNetwork(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = deque(maxlen=memory_size)

    rewards_per_episode = []
    epsilon_per_episode = []
    loss_per_episode = [0]

    step_count = 0
    best_reward = -1000

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for i in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = torch.argmax(q_network(state_tensor)).item()

            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = q_network(states).gather(1, actions).squeeze(1)
                next_q_values = target_network(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_count += 1

                # Speichere den Loss für diese Episode
                if len(loss_per_episode) <= episode:
                    loss_per_episode.append(0)  # Initialisiere Episode mit 0
                    loss_per_episode[episode-1] += loss.item()

            if done:
                break


        # Decay epsilon
        epsilon = max(epsilon - 1 / episodes, 0)
        epsilon_per_episode.append(epsilon)

        rewards_per_episode.append(total_reward)

        if episode % 1000 == 0:
            print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}")

        if total_reward >= best_reward:
            best_reward = total_reward
            print(f'Best Reward so far: {best_reward}')
            torch.save(q_network.state_dict(), f'Gradients/DQN_CartPole_{episode}.pt')

        if step_count >= network_sync_rate:
            print('Network syncing')
            target_network.load_state_dict(q_network.state_dict())
            step_count = 0

    return q_network, rewards_per_episode, epsilon_per_episode, loss_per_episode


# Test-Methode für CartPole
def test_cartpole(episodes, model_filepath):
    # CartPole Environment erstellen
    env = gym.make('CartPole-v1', render_mode='human')
    num_states = env.observation_space.shape[0]  # Dimension des Zustandsraums
    num_actions = env.action_space.n  # Anzahl möglicher Aktionen

    # Gelerntes Modell laden
    policy_dqn = QNetwork(state_size=num_states, action_size=num_actions)
    policy_dqn.load_state_dict(torch.load(model_filepath))  # Lade die gespeicherten Gewichte
    policy_dqn.eval()  # Schalte das Modell in den Evaluationsmodus

    for episode in range(episodes):
        state = env.reset()[0]  # Initialisiere die Episode
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            # Zustand vorbereiten und die beste Aktion auswählen
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Zustand in Tensor umwandeln
            with torch.no_grad():
                action = policy_dqn(state_tensor).argmax().item()  # Beste Aktion auswählen

            # Aktion ausführen
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Zustand aktualisieren
            state = next_state

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

if __name__ == '__main__':
# Training the Deep Q-Learning Agent
    env = gym.make('CartPole-v1')
    network_sync_rate = 50000
    episodes = 50000
    gamma = 0.99
    epsilon = 1.0
    max_steps = 200
    learning_rate = 0.001
    #epsilon_decay = 0.995
    buffer_size = 100000
    batch_size = 32

    #q_network, rewards_per_episode, epsilon_per_episode, loss_per_episode = deep_q_learning(env, episodes, max_steps, gamma, epsilon, buffer_size, batch_size, network_sync_rate, learning_rate)

    test_cartpole(10, "Gradients/DQN_CartPole_24811.pt")