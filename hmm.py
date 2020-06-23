import numpy as np
from random import randint

def invert(obs, obs_seq):
	n = obs.size
	t = obs_seq.size

	obs_seq_int = np.empty(dtype=int, shape=(t))
	inverse = {obs[i] : int(i) for i in range(0, n)} # remaps the values to the indicies for obs
	for i in range(t):
		obs_seq_int[i] = inverse[obs_seq[i]]
	return obs_seq_int

def generate_forwards(obs, states, init_state_prob, trans_prob, emit_prob, obs_seq):
	n = obs.size # number of observations
	k = states.size # number of states
	t = obs_seq.size # length of input sequence
	# convert observation sequence strings into numbers
	obs_seq_int = invert(obs, obs_seq)

	dp = np.zeros(dtype=float, shape=(t, k)) # forward variable
	state_seq = np.empty(dtype=int, shape=(t, k)) # prev node, used for Viterbi backtracking
	for i in range(0, k): # initialize starting probabilities
		dp[0][i] = init_state_prob[i] * emit_prob[i][obs_seq_int[0]]
		state_seq[0][i] = -1;
	for i in range(1, t):
		for j in range(0, k):
			for prev in range(0, k):
				if (dp[i][j] < dp[i-1][prev] * trans_prob[j][prev] * emit_prob[j][obs_seq_int[i]]):
					dp[i][j] = dp[i-1][prev] * trans_prob[j][prev] * emit_prob[j][obs_seq_int[i]]
					state_seq[i][j] = prev
	return dp, state_seq

def generate_backwards(obs, states, init_state_prob, trans_prob, emit_prob, obs_seq):
	n = obs.size
	k = states.size
	t = obs_seq.size
	obs_seq_int = invert(obs, obs_seq)

	dp = np.zeros(dtype=float, shape=(t, k))
	for i in range(0, k):
		dp[t-1][i] = 1
	for i in range(t-2, -1, -1):
		for j in range(0, k):
			for nex in range(0, k):
				if (dp[i][j] < dp[i+1][nex] * trans_prob[j][nex] * emit_prob[j][obs_seq_int[i]]):
					dp[i][j] = dp[i+1][nex] * trans_prob[j][nex] * emit_prob[j][obs_seq_int[i]]
	return dp

def evaluate(obs, states, init_state_prob, trans_prob, emit_prob, obs_seq):
	n = obs.size
	k = states.size
	t = obs_seq.size
	obs_seq_int = invert(obs, obs_seq)

	forwards = generate_forwards(obs, states, init_state_prob, trans_prob, emit_prob, obs_seq)[0]
	ans = 0
	for i in range(0, k):
		ans += forwards[t-1][i]
	return ans

def baum_welch(obs, states, init_state_prob, trans_prob, emit_prob, obs_seq):
	n = obs.size
	k = states.size
	t = obs_seq.size
	obs_seq_int = invert(obs, obs_seq)
	
	# variable names for arrays are taken from Rabiner paper
	alpha = generate_forwards(obs, states, init_state_prob, trans_prob, emit_prob, obs_seq)[0] # forwards variable
	beta = generate_backwards(obs, states, init_state_prob, trans_prob, emit_prob, obs_seq) # backwards variable
	print("alpha", alpha)
	print("beta", beta)
	epsilon = np.zeros(dtype=float, shape=(t, k, k)) # probability of being in state i at time t, and state j at time t+1
	for i in range(0, t-1):
		den = 0.0
		for j in range(0, k):
			for l in range(0, k):
				den = den + alpha[i][j] * beta[i][l] * trans_prob[j][l] * emit_prob[l][obs_seq_int[i+1]]
		for j in range(0, k):
			for l in range(0, k):
				epsilon[i][j][l] = alpha[i][j] * beta[i][l] * trans_prob[j][l] * emit_prob[l][obs_seq_int[i+1]] / den
	gamma = np.zeros(dtype=float, shape=(t, k)) # probability of being in state j at time i
	for i in range(0, t-1):
		for j in range(0, k):
			for l in range(0, k):
				gamma[i][j] += epsilon[i][j][l]
	print("epsilon", epsilon)
	print("gamma", gamma)	
	# reinitialize HMM parameters
	
	# new initial probabilities are set to expected frequency in state i at time t=0
	new_init = np.zeros(dtype=float, shape=(k))
	for i in range(k):
		new_init[i] = gamma[0][i]
	
	# new transition probabilies are set to expected number of transitions from i to j / expected number of transitions from i
	new_trans_prob = np.zeros(dtype=float, shape=(k, k))
	for i in range(0, k):
		for j in range(0, k):
			num = 0
			den = 0
			for l in range(0, t-1):
				num += epsilon[l][i][j]
				den += gamma[l][i]
			new_trans_prob[i][j] = num / den
	
	# new emission probabilities are set to expected number of times in i observing symbol j / expected number of times in i
	new_emit_prob = np.zeros(dtype=float, shape=(k, n))
	for i in range(0, k):
		for j in range(0, n):
			num = 0
			den = 0
			for l in range(0, t-1):
				if (obs_seq_int[l] == j):
					num += gamma[l][i]
				den += gamma[l][i]
			new_emit_prob[i][j] = num / den

	return new_init, new_trans_prob, new_emit_prob

if __name__ == "__main__":
	print("nice")
	observations = np.array(['normal', 'cold', 'dizzy'])
	states = np.array(['Healthy', 'Fever'])
	initial_state_probability = np.array([0.6, 0.4])
	transition_probabilities = np.array([ [0.7, 0.3], [0.4, 0.6] ])
	emission_probabilities = np.array([ [0.5, 0.4, 0.1], [0.1, 0.3, 0.6] ])

	observation_list = []
	for i in range(10):
		observation_list.append(observations[randint(0, 2)])
	observation_sequence = np.array(observation_list)
	print(observation_sequence)
	print("before", evaluate(observations, states, initial_state_probability, transition_probabilities, emission_probabilities, observation_sequence))
	initial_state_probability,transition_probabilities,emission_probabilities = baum_welch(observations, states, initial_state_probability, transition_probabilities, emission_probabilities, observation_sequence)
	print("init", initial_state_probability)
	print("trans", transition_probabilities)
	print("emit", emission_probabilities)
	print("after", evaluate(observations, states, initial_state_probability, transition_probabilities, emission_probabilities, observation_sequence))