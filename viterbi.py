import numpy as np
from random import randint

def viterbi(observations, states, initial_state_probability, transition_probabilities, emission_probabilities, observation_sequence):
	"""
	Runs the Viterbi algorithm for a Hidden Markov Model (HMM) with following paramters:
		observations[N] - observation space
		states[K] - hidden state space
		initial_state_probability[K] - probablity of the first state in the sequence being states[i]
		transition_probabilities[K][K] - probability of going from hidden state i to state j
		emission_prbabilities[K][N] - probability of getting observation j at state i
		observation_sequence[T] - input sequence we are trying to map to a state sequence of the same length
	Time Complexity: O(T * K^2)
	"""
	n = observations.size 				# number of observations
	k = states.size 					# number of states
	t = observation_sequence.size 		# length of input sequence
	# convert observation sequence strings into numbers
	observation_sequence_int = np.empty(dtype=int, shape=(t))
	inverse = {observations[i] : int(i) for i in range(0, n)}
	for i in range(t):
		observation_sequence_int[i] = int(inverse[observation_sequence[i]])
	state_sequence = np.empty(dtype=int, shape=(t+1, k))
	dp = np.zeros(dtype=float, shape=(t+1, k))
	for i in range(0, k): # intialize probabilities
		dp[0][i] = initial_state_probability[i]
		state_sequence[0][i] = -1;
	# compute DP transitions
	for i in range(1, t+1):
		for j in range(0, k):
			for prev in range(0, k):
				if (dp[i][j] < dp[i-1][prev] * transition_probabilities[j][prev] * emission_probabilities[j][observation_sequence_int[i-1]]):
					dp[i][j] = dp[i-1][prev] * transition_probabilities[j][prev] * emission_probabilities[j][observation_sequence_int[i-1]]
					state_sequence[i][j] = prev
	# print(dp)
	# backtrack
	end_state = (-1, -1) # probability of (state, index)
	for i in range(0, k):
		if (dp[t][i] > end_state[0]):
			end_state = (dp[t][i], i)
	answer = np.empty(dtype='U10', shape=(t))
	current_index = int(end_state[1])
	for i in range(t, 0, -1):
		# print(i, current_index)
		answer[i-1] = states[current_index]
		current_index = state_sequence[i][current_index]
	return answer

if __name__ == "__main__":
	observations = np.array(['normal', 'cold', 'dizzy'])
	states = np.array(['Healthy', 'Fever'])
	initial_state_probability = np.array([0.6, 0.4])
	transition_probabilities = np.array([ [0.7, 0.3], [0.4, 0.6] ])
	emission_probabilities = np.array([ [0.5, 0.4, 0.1], [0.1, 0.3, 0.6] ])

	observation_list = []
	for i in range(100):
		observation_list.append(observations[randint(0, 2)])
	observation_sequence = np.array(observation_list)
	print(observation_sequence)
	state_sequence = viterbi(observations, states, initial_state_probability, transition_probabilities, emission_probabilities, observation_sequence)
	print(state_sequence)