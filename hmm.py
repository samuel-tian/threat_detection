import numpy as np
from random import randint
from decimalClasses.smallDecimal import Decimal
from decimalClasses.decimalArray import DecimalArray

class HMM:
	"""class for a Hidden Markov Model"""

	def __init__(self, *args):
		if (len(args) == 5): # all parameters are initialized
			self.obs = args[0]
			self.states = args[1]
			self.init_state_prob = args[2]
			self.trans_prob = args[3]
			self.emit_prob = args[4]
		self.inverse = {self.obs[i] : int(i) for i in range(0, obs.size)}

	def __str__(self):
		ret = ""
		ret += "observations\n{}".format(self.obs)
		print("states", self.states, sep="\n")
		print("initial state probabilities", self.init_state_prob, sep="\n")
		print("transition matrix", self.trans_prob, sep="\n")
		print("emission matrix", self.emit_prob, sep="\n")
		print("------------------")
		return ret

	def invert(self, obs_seq):
		n = self.obs.size
		t = obs_seq.size

		obs_seq_int = np.empty(dtype=int, shape=(t))
		for i in range(t):
			obs_seq_int[i] = self.inverse[obs_seq[i]]
		return obs_seq_int

	def generate_forwards(self, obs_seq):
		n = self.obs.size # number of observations
		k = self.states.size # number of states
		t = obs_seq.size # length of input sequence
		# convert observation sequence strings into numbers
		obs_seq_int = self.invert(obs_seq)

		dp = DecimalArray((t, k)) # forward variable
		state_seq = np.empty(dtype=int, shape=(t, k)) # prev node, used for Viterbi backtracking
		for i in range(0, k): # initialize starting probabilities
			dp[0][i] = (self.init_state_prob[i] * self.emit_prob[i][obs_seq_int[0]])
			state_seq[0][i] = -1;
		for i in range(1, t):
			for j in range(0, k):
				for prev in range(0, k):
					if (dp[i][j] < dp[i-1][prev] * (self.trans_prob[j][prev] * self.emit_prob[j][obs_seq_int[i]])):
						dp[i][j] = dp[i-1][prev] * (self.trans_prob[j][prev] * self.emit_prob[j][obs_seq_int[i]])
						state_seq[i][j] = prev
		return dp, state_seq

	def generate_backwards(self, obs_seq):
		n = self.obs.size
		k = self.states.size
		t = obs_seq.size
		obs_seq_int = self.invert(obs_seq)

		dp = DecimalArray((t, k))
		for i in range(0, k):
			dp[t-1][i] = Decimal(1)
		for i in range(t-2, -1, -1):
			for j in range(0, k):
				for nex in range(0, k):
					if (dp[i][j] < dp[i+1][nex] * (self.trans_prob[j][nex] * self.emit_prob[j][obs_seq_int[i]])):
						dp[i][j] = dp[i+1][nex] * (self.trans_prob[j][nex] * self.emit_prob[j][obs_seq_int[i]])
		return dp

	def evaluate(self, obs_seq):
		n = self.obs.size
		k = self.states.size
		t = obs_seq.size
		obs_seq_int = self.invert(obs_seq)

		forwards = self.generate_forwards(obs_seq)[0]
		ans = Decimal(0)
		for i in range(0, k):
			ans += forwards[t-1][i]
		return ans

	def viterbi(self, obs_seq):
		n = self.obs.size
		k = self.states.size
		t = obs_seq.size
		obs_seq_int = self.invert(obs_seq)

		forwards, prev_state = self.generate_forwards(obs_seq)
		end_state = (-1, -1) # (probability, index)
		for i in range(0, k):
			if (forwards[t-1][i] > end_state[0]):
				end_state = (forwards[t-1][i], i)

		# backtrack
		answer = np.empty(dtype="U10", shape=(t))
		current_index = end_state[1]
		for i in range(t-1, -1, -1):
			answer[i] = states[current_index]
			current_index = prev_state[i][current_index]
		return answer

	def baum_welch(self, obs_seq):
		n = self.obs.size
		k = self.states.size
		t = obs_seq.size
		obs_seq_int = self.invert(obs_seq)
		
		# variable names for arrays are taken from Rabiner paper
		alpha = self.generate_forwards(obs_seq)[0] # forwards variable
		beta = self.generate_backwards(obs_seq) # backwards variable
		# print("alpha\n", alpha)
		# print("beta\n", beta)
		epsilon = DecimalArray((t, k, k)) # probability of being in state i at time t, and state j at time t+1
		for i in range(0, t-1):
			den = Decimal(0)
			for j in range(0, k):
				for l in range(0, k):
					den = den + alpha[i][j] * beta[i][l] * self.trans_prob[j][l] * self.emit_prob[l][obs_seq_int[i+1]]
			for j in range(0, k):
				for l in range(0, k):
					epsilon[i][j][l] = alpha[i][j] * beta[i][l] * self.trans_prob[j][l] * self.emit_prob[l][obs_seq_int[i+1]] / den
		gamma = DecimalArray((t, k)) # probability of being in state j at time i
		for i in range(0, t-1):
			for j in range(0, k):
				for l in range(0, k):
					gamma[i][j] += epsilon[i][j][l]

		# reinitialize HMM parameters
		
		# new initial probabilities are set to expected frequency in state i at time t=0
		new_init = DecimalArray((k))
		for i in range(k):
			new_init[i] = gamma[0][i]
		
		# new transition probabilies are set to expected number of transitions from i to j / expected number of transitions from i
		new_trans_prob = DecimalArray((k, k))
		for i in range(0, k):
			for j in range(0, k):
				num = Decimal(0)
				den = Decimal(0)
				for l in range(0, t-1):
					num += epsilon[l][i][j]
					den += gamma[l][i]
				new_trans_prob[i][j] = num / den
		
		# new emission probabilities are set to expected number of times in i observing symbol j / expected number of times in i
		new_emit_prob = DecimalArray((k, n))
		for i in range(0, k):
			for j in range(0, n):
				num = Decimal(0)
				den = Decimal(0)
				for l in range(0, t-1):
					if (obs_seq_int[l] == j):
						num += gamma[l][i]
					den += gamma[l][i]
				new_emit_prob[i][j] = num / den

		self.init_state_prob = new_init
		self.trans_prob = new_trans_prob
		self.emit_prob = new_emit_prob

if __name__ == "__main__":
	obs = np.array(['normal', 'cold', 'dizzy'])
	states = np.array(['Healthy', 'Fever'])
	init_state_prob = DecimalArray([0.6, 0.4])
	trans_prob = DecimalArray([ [0.7, 0.3], [0.4, 0.6] ])
	emit_prob = DecimalArray([ [0.5, 0.4, 0.1], [0.1, 0.3, 0.6] ])

	observation_list = []
	for i in range(100):
		observation_list.append(obs[randint(0, 2)])
	observation_sequence = np.array(observation_list)
	print(observation_sequence)
	
	identifier = HMM(obs, states, init_state_prob, trans_prob, emit_prob)
	print("before", identifier.evaluate(observation_sequence))

	state_seq = identifier.viterbi(observation_sequence)
	# print(state_seq)

	print(identifier)

	print("----------------------")	

	identifier.baum_welch(observation_sequence)
	print("after", identifier.evaluate(observation_sequence))

	state_seq = identifier.viterbi(observation_sequence)
	# print(state_seq)

	print(identifier)
