import timeit
import os, sys
import math
import numpy as np
from random import randint
# from arithmeticClasses.extendedExp import ExtendedExp as EE
from HMMClasses.arithmeticClasses.extendedExp import ExtendedExp as EE

class HMMScaling:

	"""Implementation of a Hidden Markov Model with ExtendedExp scaling to avoid precision errors
	HMM implementation is only intended to work on a single input sequence

	n: number of states
	k: number of observations
	t: length of observation sequence

	all following paramters are numpy arrays
	obs[k]: array of observations
	states[n]: array of states
	init_prob[n]: probability of being in state i initially
	trans_prob[n][n]: probability of transitioning from state i to state j
	emit_prob[n][k]: probability of emitting observation j in state i

	main functions:
	evaluate(self, observation_sequence) -> float
		determines the probability of observing the input sequence, with the current model paramters
		in this case, the return value is the natural logarithm of the actual probability
	baum_welch(self, observation_sequence) -> null
		updates the current model in order to maximize the probability of observing the input sequence
		this is the training aspect of the HMM
	viterbi(self, observation_sequence) -> np.array (NOT IMPLEMENTED YET)
		returns the state sequence that has the highest probability of generating the input sequence
		we won't be using this much, unless we are trying to discover some underlying structure of chase trajectories, because the states we use for our HMMs are arbitrary

	Usage:
		instantiate HMM with default paramters - model = HMMScaling(number of hidden states, number of observations)
		train HMM with input sequence - model.baum_welch(observation_sequence)
		evaluate probability of observing given sequence - model.evaluate(observation_sequence)

		When classifying marine activity, create a separate HMM for each activity i.e. HMM for normal, HMM for following, HMM for circling, etc.
		Train each HMM with corresponding trajectories i.e. train normal HMM with normal trajectories, following with following trajectories, etc.
		For each unidentified input sequence, evaluate the probability of observing that sequence for all of the HMMs. Determine which HMM returns the highest probability, and then select that activity.
	"""

	def __init__(self, *args):
		"""Initializes HMM

		args[5]: (obs, states, initial state prob, transition prob, emit prob)
		args[1]: (emit prob)
		args[2]: (number of states, number of coordinates)
		"""
		if (len(args)==5): # all paramters are defined
			self.obs = args[0]
			self.states = args[1]
			self.init_prob = args[2]
			self.trans_prob = args[3]
			self.emit_prob = args[4]
		else:
			n = 0 # number of states
			k = 0 # number of observations
			if (len(args)==2): # number of states and number of coordinates are provided, all other paramters are uniform
				n = args[0]
				k = args[1] * args[1] # tuples of coordinates should be mapped to a single integer
				self.emit_prob = np.empty(dtype=float, shape=(n, k))
				# initialize emission probability matrix to uniform distribution
				for i in range(n):
					for j in range(k):
						self.emit_prob[i][j] = EE.elnproduct(EE.eln(1), -EE.eln(k))
			else: # emission matrix is provided, all other paramters are uniform
				n = args[0].shape[0]
				k = args[0].shape[1]
				self.emit_prob = np.empty(dtype=float, shape=(n, k))
				for i in range(n):
					for j in range(k):
						self.emit_prob[i][j] = EE.eln(args[0][i][j])
			# initialize observation array
			self.obs = np.empty(dtype=int, shape=(k))
			for i in range(k):
				self.obs[i] = i
			# initialize state array
			self.states = np.empty(dtype=int, shape=(n))
			for i in range(n):
				self.states[i] = i
			# initialize inital state probability array to uniform distribution
			self.init_prob = np.empty(dtype=float, shape=(n))
			for i in range(n):
				self.init_prob[i] = EE.eln(1 / n)
			# initialize transition probability matrix to uniform distribution
			self.trans_prob = np.empty(dtype=float, shape=(n, n))
			for i in range(n):
				for j in range(n):
					self.trans_prob[i][j] = EE.eln(1 / n)

	def __str__(self):
		"""Displays obs, states, initial state prob, transition prob, and emit prob"""
		ret = ""
		ret += "observations: " + str(self.obs) + "\n"
		ret += "states: " + str(self.states) + "\n"
		ret += "initial state prob: " + str(self.init_prob) + "\n"
		ret += "transition matrix\n" + str(self.trans_prob) + "\n"
		ret += "emission matrix\n" + str(self.emit_prob) + "\n"
		return ret

	def generate_forwards(self, obs_seq):
		"""Calculates forward variable, also known as alpha[T][N]

		alpha[i][j] = probability of the partial observation sequence obs_seq[0...i] where the current state is j
		Memory Complexity: O(T * N)
		Time Complexity: O(T * N^2)
		"""
		n = self.states.size
		k = self.obs.size
		t = obs_seq.size # length of the given observed sequence

		"""forward variable dynamic programming dp[t][n]
		states: dp[i][j] = probability of observing the partial observation sequence obs_seq[0...i] with current state j
		initialize: dp[0][i] = init_prob[i] * emit_prob[i][obs_seq[0]]
		transitions: dp[i][j] = emit_prob[j][obs_seq[i]] * sum_{prev} (dp[i-1][prev] * trans_prob[prev][j])
		"""
		dp = np.empty(dtype=float, shape=(t, n))
		for i in range(n):
			dp[0][i] = EE.elnproduct(self.init_prob[i], self.emit_prob[i][obs_seq[0]])
		for i in range(1, t):
			for j in range(n):
				logalpha = float('nan')
				for prev in range(n):
					logalpha = EE.elnsum(logalpha, EE.elnproduct(dp[i-1][prev], self.trans_prob[prev][j]))
				dp[i][j] = EE.elnproduct(logalpha, self.emit_prob[j][obs_seq[i]])
		return dp

	def generate_backwards(self, obs_seq):
		"""Calculates backward variable, also known as beta[T][N]

		beta[i][j] = probability of the partial observation sequence obs_seq[i...t-1] where the current state is j
		Memory Complexity: O(T * N)
		Time Complexity: O(T * N^2)
		"""
		n = self.states.size
		k = self.obs.size
		t = obs_seq.size

		"""backward variable dynamic programming dp[t][n]
		states[i][j] = probability of observing the partial observation sequence obs_seq[i...t-1] with current state j
		initialize: dp[t-1][i] = 1
		transitions: dp[i][j] = sum_{next} (emit_prob[next][obs_seq[i+1]] * dp[i+1][next] * trans_prob[j][next])
		"""
		dp = np.empty(dtype=float, shape=(t, n))
		for i in range(n):
			dp[t-1][i] = 0 # == eln(1)
		for i in range(t-2, -1, -1):
			for j in range(n):
				logbeta = float('nan')
				for nex in range(n):
					logbeta = EE.elnsum(logbeta, EE.elnproduct(self.trans_prob[j][nex], EE.elnproduct(self.emit_prob[nex][obs_seq[i+1]], dp[i+1][j])))
				dp[i][j] = logbeta
		return dp

	def evaluate(self, obs_seq):
		"""Determines the probability of seeing obs_seq given the current model paramters

		Memory Complexity: O(T * N)
		Time Complexity: O(T * N^2)
		"""
		n = self.states.size
		k = self.obs.size
		t = obs_seq.size

		alpha = self.generate_forwards(obs_seq)
		# evaluation probability is the sum over all alpha at T=t-1
		eval_value = float('nan')
		for i in range(n):
			eval_value = EE.elnsum(eval_value, alpha[t-1][i])
		return eval_value

	def generate_gamma(self, obs_seq, alpha, beta):
		"""Calculates gamma[T][N]

		gamma[i][j] = probability of being in state j at time i
		Memory Complexity: O(T * N)
		Time Complexity: O(T * N)
		"""
		n = self.states.size
		k = self.obs.size
		t = obs_seq.size

		"""gamma variable calculation
		gamma[i][j] = alpha[i][j] * beta[i][j] / (sum over all alpha[x][y]*beta[x][y])
		"""
		gamma = np.empty(dtype=float, shape=(t, n))
		for i in range(t):
			normalizer = float('nan')
			for j in range(n):
				gamma[i][j] = EE.elnproduct(alpha[i][j], beta[i][j])
				normalizer = EE.elnsum(normalizer, gamma[i][j])
			for j in range(n):
				gamma[i][j] = EE.elnproduct(gamma[i][j], -normalizer)
		return gamma

	def generate_epsilon(self, obs_seq, alpha, beta):
		"""Calculates epsilon[T][N][N]

		epsilon[i][j][k] = probability of probability of being in state j at time i and state k at time i+1
		Memory Complexity: O(T * N^2)
		Time Complexity: O(T * N^2)
		"""
		n = self.states.size
		k = self.obs.size
		t = obs_seq.size

		epsilon = np.empty(dtype=float, shape=(t, n, n))
		for i in range(t-1):
			normalizer = float('nan')
			for j in range(n):
				for l in range(n):
					epsilon[i][j][l] = EE.elnproduct(alpha[i][j], EE.elnproduct(self.trans_prob[j][l],
						EE.elnproduct(self.emit_prob[l][obs_seq[i+1]], beta[i+1][l])))
					normalizer = EE.elnsum(normalizer, epsilon[i][j][l])
			for j in range(n):
				for l in range(n):
					epsilon[i][j][l] = EE.elnproduct(epsilon[i][j][l], -normalizer)
		return epsilon

	def baum_welch(self, obs_seq):
		"""Updates the paramters of the model to maximize the probability of seeing the sequence obs_seq

		Memory Complexity: O(T * N^2)
		Time Complexity: O(T * N^2 + N * K (*T, can be optimized out))
		"""
		n = self.states.size
		k = self.obs.size
		t = obs_seq.size

		alpha = self.generate_forwards(obs_seq)
		beta = self.generate_backwards(obs_seq)
		gamma = self.generate_gamma(obs_seq, alpha, beta)
		epsilon = self.generate_epsilon(obs_seq, alpha, beta)

		# re-estimate initial state probabilities
		for i in range(n):
			self.init_prob[i] = gamma[0][i]

		# re-estimate transition matrix
		for i in range(n):
			for j in range(n):
				num = float('nan')
				den = float('nan')
				for l in range(t-1):
					num = EE.elnsum(num, epsilon[l][i][j])
					den = EE.elnsum(den, gamma[l][i])
				self.trans_prob[i][j] = EE.elnproduct(num, -den)

		# re-estimate emission matrix
		adj = []
		for j in range(k):
			tmp = []
			for l in range(t):
				if (j == obs_seq[l]):
					tmp.append(l)
			adj.append(tmp)
		for i in range(n):
			den = float('nan')
			for l in range(t):
				den = EE.elnsum(den, gamma[l][i])
			for j in range(k):
				num = float('nan')
				for l in adj[j]:
					num = EE.elnsum(num, gamma[l][i])
				self.emit_prob[i][j] = EE.elnproduct(num, -den)

if __name__ == "__main__":
	n = 5
	k = 5
	emit_prob = np.empty(dtype=float, shape=(n, k*k))
	for i in range(n): cur = 1
		for j in range(k*k-1):
			x = randint(1, 9)
			emit_prob[i][j] = cur * x / 10
			cur -= cur * x / 10
		emit_prob[i][k*k-1] = cur
	print(emit_prob)
	identifier = HMMScaling(emit_prob)

	tmp = []
	for i in range(15):
		tmp.append(k * randint(0, k-1) + randint(0, k-1))
	obs_seq = np.empty(shape=len(tmp), dtype='O')
	obs_seq[:] = tmp
	print(obs_seq)

	print(identifier)
	print(identifier.evaluate(obs_seq))

	for i in range(100):
		identifier.baum_welch(obs_seq)
		# print(identifier)
		print(identifier.evaluate(obs_seq))

	print(identifier)
