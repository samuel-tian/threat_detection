#include <iostream>
#include <gmp.h>
#include <gmpxx.h>
#include <math.h>
#include <random>
#include <chrono>

#include "hmm.h"

/**
 * redefining a vector of vectors as a matrix, for readability
 */
template<typename T>
using matrix = std::vector<std::vector<T> >;

/**
 * Initialize HMM with a set number of states and emissions.
 * Vectors and matrices will be initialized to a uniform distribution.
 *
 * @param num_states the number of states in the HMM
 * @param num_emissions the number of emissions in the HMM
 */
HMM::HMM(int num_states, int num_emissions) {
	this->N = num_states;
	this->K = num_emissions;
	int n = this->N;
	int k = this->K;
	mpf_class x = n;
	x = 1 / x;
	// initialize initial state probabilities to uniform distribution
	// p[i] = 1 / n
	this->init_prob.resize(n);
	for (int i = 0; i < n; i++) {
		init_prob[i] = x;
	}
	// initialize transition probabilties to uniform distribution
	// p[i][j] = 1 / n
	this->trans_prob.resize(n, std::vector<mpf_class>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			this->trans_prob[i][j] = x;
		}
	}	
	// initialize emission probabilities to uniform distribution
	// p[i][j] = 1 / k
	x = k;
	x = 1 / x;
	this->emit_prob.resize(n, std::vector<mpf_class>(k));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			this->emit_prob[i][j] = x;
		}
	}
}

/**
 * Initialize HMM with an emissions probability matrix.
 * Transition and initial probability matrices will be initialized to a uniform distribution
 *
 * @param emit_prob emission matrix with dimensions n x k
 */
HMM::HMM(matrix<mpf_class>& emit_prob) {
	this->N = emit_prob.size();
	this->K = emit_prob[0].size();
	int n = this->N;
	int k = this->K;
	mpf_class x = n;
	x = 1 / x;
	// intialize initial state probabilites to uniform distribution
	// p[i] = 1 / n
	this->init_prob.resize(n);
	for (int i = 0; i < n; i++) {
		this->init_prob[i] = x;
	}
	// initialize transition probabilities to uniform distribution
	// p[i][j] = 1 / n
	this->trans_prob.resize(n, std::vector<mpf_class>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			this->trans_prob[i][j] = x;
		}
	}
	// initialize emission probabilities to specified matrix
	this->emit_prob = emit_prob;
}

/**
 * Enable HMM class to be passed to an outputstream
 *
 * @param os outputstream that HMM should be passed to
 * @param hmm HMM object that is being passed to outputstream
 * @return os outputstream that was used
 */
std::ostream& operator << (std::ostream& os, const HMM& hmm) {
	int n = hmm.N;
	int k = hmm.K;
	os << "number of states: " << n << "\n";
	os << "number of emissions: " << k << "\n";
	os << "initial state probabilities: ";
	for (int i = 0; i < n; i++) {
		os << hmm.init_prob[i] << " ";
	}
	os << "\n";
	os << "transition probabilities" << "\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			os << hmm.trans_prob[i][j] << " ";
		}
		os << "\n";
	}
	os << "emission probabilities" << "\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			os << hmm.emit_prob[i][j] << " ";
		}
		os << "\n";
	}
	return os;
}

/**
 * Generate scaled forward variable, as specified in Rabiner's paper
 * Unscaled forward variable is defined by the recursion:
 * 1) if t=1, a(t)(n) = init_prob[n] * emit_prob[n][obs_seq[0]]
 * 2) else, a(t)(n) = emit_prob[n][obs_seq[t]] * sum_previous{trans_prob[previous][n] * a(t-1)(previous)}
 * Scaled forward variable rescales all values at time t to sum to 1
 * a'(t)(n) = a(t)(n) / sum_i{a(t)(i)}
 *
 * Memory: alpha[T][N]
 * Time Complexity: O(T * N^2)
 *
 * @param obs_seq observation sequence that will be used to generate the forwards recursion
 * @return pair of vectors, containing
 * 1) the scaled forward variables, a matrix of dimension scaled[T][N]
 * 2) the scale factors, a vector of dimension scale_factors[T]
 */
std::pair<matrix<mpf_class>, std::vector<mpf_class> > HMM::generate_forwards(std::vector<int>& obs_seq) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	matrix<mpf_class> unscaled(t, std::vector<mpf_class>(n));
	matrix<mpf_class> scaled(t, std::vector<mpf_class>(n));
	std::vector<mpf_class> scale_factors(t);
	for (int i = 0; i < t; i++) {
		// base case
		if (i == 0) {
			for (int j = 0; j < n; j++) {
				unscaled[i][j] = this->init_prob[j] * this->emit_prob[j][obs_seq[i]];
			}
			scale_factors[i] = 1;
		}
		// inductive step
		else {
			mpf_class alphasum = 0;
			for (int j = 0; j < n; j++) {
				mpf_class alpha = 0;
				for (int prev = 0; prev < n; prev++) {
					alpha += scaled[i-1][prev] * this->trans_prob[prev][j];
				}
				alpha *= this->emit_prob[j][obs_seq[i]];
				unscaled[i][j] = alpha;
				alphasum += alpha;
			}
			scale_factors[i] = 1 / alphasum;
		}
		// scale all forward values
		for (int j = 0; j < n; j++) {
			scaled[i][j] = scale_factors[i] * unscaled[i][j];
		}
	}

	return std::make_pair(scaled, scale_factors);
}

/**
 * Generate scaled backward variable, as specified in Rabiner's paper
 * Unscaled backward variable is defined by the recusion:
 * 1) it t=T, b(t)(n) = 1
 * 2) else, b(t)(n) = sum_next{emit_prob[next][obs_seq[i+1]] * a(i+1)(next) * trans_prob[j][next]}
 * Scaled backward variable uses the scale factors from the forward recursion, as the forwards and backwards variables will be of similar magnitude
 * b'(t)(n) = b(t)(n) * scale_factor[t]
 *
 * Memory: beta[T][N]
 * Time Complexity: O(T * N^2)
 *
 * @param obs_seq observation sequence that will be used to generate the backwards recursion
 * @param scale_factor set of scale factors that were generated by the forwards recursion, which will be used to scale the backwards variable
 * @return the scaled backward variables, a matrix of dimension scaled[T][N]
 */
matrix<mpf_class> HMM::generate_backwards(std::vector<int>& obs_seq, std::vector<mpf_class>& scale_factors) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	matrix<mpf_class> unscaled(t, std::vector<mpf_class>(n));
	matrix<mpf_class> scaled(t, std::vector<mpf_class>(n));
	for (int i = t-1; i >= 0; i--) {
		// base case
		if (i == t-1) {
			for (int j = 0; j < n; j++) {
				unscaled[i][j] = 1;
			}
		}
		// induction
		else {
			for (int j = 0; j < n; j++) {
				mpf_class beta = 0;
				for (int nex = 0; nex < n; nex++) {
					beta += this->trans_prob[j][nex] * this->emit_prob[nex][obs_seq[i+1]] * scaled[i+1][j];
				}
				unscaled[i][j] = beta;
			}
		}
		// scale all backward values using scale_factors
		for (int j = 0; j < n; j++) {
			scaled[i][j] = unscaled[i][j] * scale_factors[i];
		}
	}

	return scaled;
}

/**
 * Finds the probability of observing an observation sequence, given the current model parameters
 * Sums the forward variables at time T over all potential ending states
 * 
 * Time Complexity: O(T * N^2)
 * 
 * @param obs_seq observation sequence
 * @return logarithm of the probability of seeing the given observation sequence, because the actual probability is likely to underflow, even with the arbitrary precision library
 */
mpf_class HMM::evaluate(std::vector<int>& obs_seq) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	std::vector<mpf_class> scale_factors = (this->generate_forwards(obs_seq)).second;
	mpf_class eval_value = 1;
	for (int i = 0; i < t; i++) {
		eval_value /= scale_factors[i];
	}
	return eval_value;
}

/**
 * Generates gamma variable, as specified in Rabiner's paper
 * Gamma is defined as the probability of being in a certain state at a certain time
 * gamma[i][j] = alpha[i][j] * beta[i][j] / scale_factors[i]
 *
 * Memory: gamma[T][N]
 * Time Complexity: O(T * N)
 *
 * @param obs_seq vector of observations
 * @param alpha scaled forward variables for given observation sequence
 * @param beta scaled backward variables for given observation sequence
 * @param scale_factors scale factors used to scale forward variables
 * @return the scaled gamma variables
 */
matrix<mpf_class> HMM::generate_gamma(std::vector<int>& obs_seq, matrix<mpf_class>& alpha, matrix<mpf_class>& beta, std::vector<mpf_class>& scale_factors) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	matrix<mpf_class> gamma(t, std::vector<mpf_class>(n));
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < n; j++) {
			gamma[i][j] = alpha[i][j] * beta[i][j] / scale_factors[i];
		}
	}
	return gamma;
}

/**
 * Generates epsilon variable, as specified in Rabiner's paper
 * Epsilon is defined as the probability of being in a certain state at time t and another state at time t+1
 * epsilon[i][j][k] = alpha[i][j] * beta[i+1][k] * trans_prob[j][k] * emit_prob[j][obs_seq[i]]
 *
 * Memory: epsilon[T][N][N]
 * Time Complexity: O(T * N^2)
 *
 * @param obs_seq vector of observations
 * @param alpha scaled forward variables for given observation sequence
 * @param beta scaled backward variables for given observation sequence
 * @param scale_factors scale factors used to scale forward variables
 * @return the scaled epsilon variables
 */
std::vector<matrix<mpf_class> > HMM::generate_epsilon(std::vector<int>& obs_seq, matrix<mpf_class>& alpha, matrix<mpf_class>& beta, std::vector<mpf_class>& scale_factors) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	std::vector<matrix<mpf_class> > epsilon(t, matrix<mpf_class>(n, std::vector<mpf_class>(n)));
	for (int i = 0; i < t-1; i++) {
		mpf_class normalizer = 0;
		for (int j = 0; j < n; j++) {
			for (int l = 0; l < n; l++) {
				epsilon[i][j][l] = alpha[i][j] * beta[i+1][l] * this->trans_prob[j][l] * this->emit_prob[l][obs_seq[i+1]];
			}
		}
	}
	return epsilon;
}

/**
 * Re-estimates pi, A, B parameters according to the Baum Welch Algorithm outlined in Rabiner's paper
 * pi is reestimated as the expected frequency in state i at time t=0
 * A is reestimated as the expected number of transitions from state i to state j, divided by the expected number of transitions from state i
 * B is reestimated as the expected number of times in state j and observing symbol obs_seq[i], divided by the expected number of times in state i
 *
 * Memory: no additional memory asides from auxillary functions
 * Time Complexity: O(T * N^2 + N * K)
 *
 * @param obs_seq vector of observations
 * @return void
 */
void HMM::baum_welch(std::vector<int>& obs_seq) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	std::pair<matrix<mpf_class>, std::vector<mpf_class> > alpha_scale_pair = this->generate_forwards(obs_seq);
	matrix<mpf_class> alpha = alpha_scale_pair.first;
	std::vector<mpf_class> scale_factors = alpha_scale_pair.second;
	matrix<mpf_class> beta = this->generate_backwards(obs_seq, scale_factors);
	matrix<mpf_class> gamma = this->generate_gamma(obs_seq, alpha, beta, scale_factors);
	std::vector<matrix<mpf_class> > epsilon = this->generate_epsilon(obs_seq, alpha, beta, scale_factors);

	// re-estimate initial state probabilities
	for (int i = 0; i < n; i++) {
		this->init_prob[i] = gamma[0][i];
	}
	// re-estimate transition matrix probabilities
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			mpf_class num = 0, den = 0;
			for (int l = 0; l < t-1; l++) {
				num += epsilon[l][i][j];
				den += gamma[l][i];
			}
			this->trans_prob[i][j] = num / den;
		}
	}
	// re-estimate emission matrix probabilities
	std::vector<std::vector<int> > obs_seq_vals(k); // obs_seq_vals[i] contains a list of indicies of obs_seq that have value equal to i
	for (int i = 0; i < t; i++) {
		obs_seq_vals[obs_seq[i]].push_back(i);
	}
	for (int i = 0; i < n; i++) {
		mpf_class den = 0;
		for (int j = 0; j < t; j++) {
			den += gamma[j][i];
		}
		for (int j = 0; j < k; j++) {
			mpf_class num = 0;
			for (int l : obs_seq_vals[j]) {
				num += gamma[l][i];
			}
			this->emit_prob[i][j] = num / den;
		}
	}
}

/**
 * Re-estimates A, B paramters according to the Baum Welch Algorithm outlined in Rabiner's paper
 * Allows for multiple input observation sequences
 *
 * Memory: A[n][n], B[n][k]
 * Time Complexity: O(X * (T * N^2 + N * K)), where X is the number of training sequences
 * 
 * @param obs_seq vector of observation sequences
 * @return void
 */
void HMM::multi_baum_welch(std::vector<std::vector<int> >& obs_seqs) {
	int n = this->N;
	int k = this->K;
	matrix<mpf_class> a_num(n, std::vector<mpf_class>(n));
	matrix<mpf_class> a_den(n, std::vector<mpf_class>(n));
	matrix<mpf_class> b_num(n, std::vector<mpf_class>(k));
	matrix<mpf_class> b_den(n, std::vector<mpf_class>(k));
	for (std::vector<int>& obs_seq : obs_seqs) {
		int t = obs_seq.size();
		// initialize auxillary variables
		std::pair<matrix<mpf_class>, std::vector<mpf_class> > alpha_scale_pair = this->generate_forwards(obs_seq);
		matrix<mpf_class> alpha = alpha_scale_pair.first;
		std::vector<mpf_class> scale_factors = alpha_scale_pair.second;
		matrix<mpf_class> beta = this->generate_backwards(obs_seq, scale_factors);
		matrix<mpf_class> gamma = this->generate_gamma(obs_seq, alpha, beta, scale_factors);
		std::vector<matrix<mpf_class> > epsilon = this->generate_epsilon(obs_seq, alpha, beta, scale_factors);
		// re-estimate transition matrix probabilities
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				for (int l = 0; l < t-1; l++) {
					a_num[i][j] += epsilon[l][i][j];
					a_den[i][j] += gamma[l][i];
				}
			}
		}
		// re-estimate emission matrix probabilities
		std::vector<std::vector<int> > obs_seq_vals(k);
		for (int i = 0; i < t; i++) {
			obs_seq_vals[obs_seq[i]].push_back(i);
		}
		for (int i = 0; i < n; i++) {
			mpf_class den = 0;
			for (int j = 0; j < t; j++) {
				den += gamma[j][i];
			}
			for (int j = 0; j < k; j++) {
				for (int l : obs_seq_vals[j]) {
					b_num[i][j] += gamma[l][i];
				}
				b_den[i][j] += den;
			}
		}
	}
	// update transition probability matrix
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			this->trans_prob[i][j] = a_num[i][j] / a_den[i][j];
		}
	}
	// update emission probability matrix
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			this->emit_prob[i][j] = b_num[i][j] / b_den[i][j];
		}
	}
}

int main() {
	mpf_set_default_prec(256);
	std::cout << "precision: " << mpf_get_default_prec() << "\n";
	int n = 5;
	int k = 25;
	int t = 100;
	matrix<mpf_class> emit_prob;

	//  Mersene twister, which is a random number generator that has better performance than C++ rand() / srand()
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);
	// generate emission matrix
	emit_prob.resize(n, std::vector<mpf_class>(k));
	for (int i = 0; i < n; i++) {
		int total = 0;
		for (int j = 0; j < k; j++) {
			int val = 1 + generator() % 9; // generate a random value from 1 to 9
			total += val;
			emit_prob[i][j] = val;
		}
		// scale all values in the emission matrix to [0, 1]
		for (int j = 0; j < k; j++) {
			emit_prob[i][j] = emit_prob[i][j] / total;
		}
	}
	HMM hmm(emit_prob);

	std::vector<std::vector<int> > obs_seqs;
	int x = 15;
	obs_seqs.resize(15);
	for (int i = 0; i < x; i++) {
		obs_seqs[i].resize(t);
		for (int j = 0; j < t; j++) {
			obs_seqs[i][j] = generator() % k;
		}	
	}
	// checking how long each iteration for training and evaluation takes
	std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now(); // time before execution

	std::cout << "BEFORE\n";
	for (int i = 0; i < x; i++) {
		std::cout << hmm.evaluate(obs_seqs[i]) << "\n";
	}
	hmm.multi_baum_welch(obs_seqs);
	std::cout << "AFTER\n";
	for (int i = 0; i < x; i++) {
		std::cout << hmm.evaluate(obs_seqs[i]) << "\n";
	}

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); // time after execution
	std::cout << "TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << "\n"; // time elapsed

	return 0;
}