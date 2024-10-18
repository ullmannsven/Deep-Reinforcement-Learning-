import numpy as np 

def k_armed_bandit(k, num_of_iter, eps, means, std_dev):
    all_values = np.zeros((k, num_of_iter))
    no_of_samples = np.zeros(k)
    total_reward = np.zeros(num_of_iter)
    total_regret = np.zeros(num_of_iter)

    for i in range(k):
        all_values[i][0] = np.random.normal(loc=means[i], scale=std_dev[i])
        no_of_samples[i] += 1

    total_reward[0] = np.sum(all_values[0, :], axis=0)

    for i in range(1, num_of_iter):
        eps_yes_no = np.random.uniform(0,1)
        if eps_yes_no <= eps: 
            current_bandit = int(np.floor(np.random.uniform(0,1) * k))
            all_values[current_bandit, i] = np.random.normal(loc=means[current_bandit], scale=std_dev[current_bandit])
        else: 
            current_bandit = np.argmax(np.sum(all_values[:, :i], axis=1) / no_of_samples)
            no_of_samples[current_bandit] += 1
            all_values[current_bandit, i] = np.random.normal(loc=means[current_bandit], scale=std_dev[current_bandit])
        
        total_reward[i] = total_reward[i-1] + all_values[current_bandit, i]

    print(total_reward)

    return all_values, total_reward













