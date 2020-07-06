#include <iostream>
#include <gmp.h>
#include <gmpxx.h>
#include <math.h>
#include <random>
#include <chrono>
#include <climits>

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
	mpfr::mpreal x = n;
	x = 1 / x;
	// initialize initial state probabilities to uniform distribution
	// p[i] = 1 / n
	this->init_prob.resize(n);
	for (int i = 0; i < n; i++) {
		init_prob[i] = x;
	}
	// initialize transition probabilties to uniform distribution
	// p[i][j] = 1 / n
	this->trans_prob.resize(n, std::vector<mpfr::mpreal>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			this->trans_prob[i][j] = x;
		}
	}	
	// initialize emission probabilities to uniform distribution
	// p[i][j] = 1 / k
	x = k;
	x = 1 / x;
	this->emit_prob.resize(n, std::vector<mpfr::mpreal>(k));
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
HMM::HMM(matrix<mpfr::mpreal>& emit_prob) {
	this->N = emit_prob.size();
	this->K = emit_prob[0].size();
	int n = this->N;
	int k = this->K;
	mpfr::mpreal x = n;
	x = 1 / x;
	// intialize initial state probabilites to uniform distribution
	// p[i] = 1 / n
	this->init_prob.resize(n);
	for (int i = 0; i < n; i++) {
		this->init_prob[i] = x;
	}
	// initialize transition probabilities to uniform distribution
	// p[i][j] = 1 / n
	this->trans_prob.resize(n, std::vector<mpfr::mpreal>(n));
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
	mpfr::mpreal total = 0;
	os << "number of states: " << n << "\n";
	os << "number of emissions: " << k << "\n";
	os << "initial state probabilities: ";
	total = 0;
	for (int i = 0; i < n; i++) {
		os << hmm.init_prob[i] << " ";
		total += hmm.init_prob[i];
	}
	os << "-> " << total << "\n";
	os << "transition probabilities" << "\n";
	for (int i = 0; i < n; i++) {
		total = 0;
		for (int j = 0; j < n; j++) {
			os << hmm.trans_prob[i][j] << " ";
			total += hmm.trans_prob[i][j];
		}
		os << "-> " << total << " ";
		os << "\n";
	}
	os << "emission probabilities" << "\n";
	for (int i = 0; i < n; i++) {
		total = 0;
		for (int j = 0; j < k; j++) {
			os << hmm.emit_prob[i][j] << " ";
			total += hmm.emit_prob[i][j];
		}
		os << "-> " << total << " ";
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
std::pair<matrix<mpfr::mpreal>, std::vector<mpfr::mpreal> > HMM::generate_forwards(std::vector<int>& obs_seq) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	matrix<mpfr::mpreal> unscaled(t, std::vector<mpfr::mpreal>(n));
	matrix<mpfr::mpreal> scaled(t, std::vector<mpfr::mpreal>(n));
	std::vector<mpfr::mpreal> scale_factors(t);
	for (int i = 0; i < t; i++) {
		// base case
		if (i == 0) {
			mpfr::mpreal initsum = 0;
			for (int j = 0; j < n; j++) {
				unscaled[i][j] = this->init_prob[j] * this->emit_prob[j][obs_seq[i]];
				initsum += unscaled[i][j];
			}
			scale_factors[i] = 1 / initsum;
		}
		// inductive step
		else {
			mpfr::mpreal alphasum = 0;
			for (int j = 0; j < n; j++) {
				mpfr::mpreal alpha = 0;
				for (int prev = 0; prev < n; prev++) {
					alpha += scaled[i-1][prev] * this->trans_prob[prev][j];
				}
				unscaled[i][j] = alpha * this->emit_prob[j][obs_seq[i]];
				alphasum += unscaled[i][j];
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
matrix<mpfr::mpreal> HMM::generate_backwards(std::vector<int>& obs_seq, std::vector<mpfr::mpreal>& scale_factors) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	matrix<mpfr::mpreal> scaled(t, std::vector<mpfr::mpreal>(n));
	for (int i = t-1; i >= 0; i--) {
		// base case
		if (i == t-1) {
			for (int j = 0; j < n; j++) {
				scaled[i][j] = scale_factors[i];
			}
		}
		// induction
		else {
			for (int j = 0; j < n; j++) {
				mpfr::mpreal beta = 0;
				for (int nex = 0; nex < n; nex++) {
					beta += this->trans_prob[j][nex] * this->emit_prob[nex][obs_seq[i+1]] * scaled[i+1][j];
				}
				scaled[i][j] = beta * scale_factors[i];
			}
		}
	}

	return scaled;
}

/**
 * Finds the logarithm of the probability of observing an observation sequence, given the current model parameters
 * Sums the forward variables at time T over all potential ending states
 * 
 * Time Complexity: O(T * N^2)
 * 
 * @param obs_seq observation sequence
 * @return logarithm of the probability of seeing the given observation sequence, because the actual probability is likely to underflow, even with the arbitrary precision library
 */
mpfr::mpreal HMM::evaluate(std::vector<int>& obs_seq) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	std::vector<mpfr::mpreal> scale_factors = (this->generate_forwards(obs_seq)).second;
	mpfr::mpreal eval_value = 0;
	for (int i = 0; i < t; i++) {
		eval_value += -log(scale_factors[i]);
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
matrix<mpfr::mpreal> HMM::generate_gamma(std::vector<int>& obs_seq, std::vector<matrix<mpfr::mpreal> > epsilon, std::vector<mpfr::mpreal>& scale_factors) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	matrix<mpfr::mpreal> gamma(t, std::vector<mpfr::mpreal>(n));
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < n; j++) {
			mpfr::mpreal gammasum = 0;
			for (int l = 0; l < n; l++) {
				gammasum += epsilon[i][j][l];
			}
			gamma[i][j] = gammasum;
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
std::vector<matrix<mpfr::mpreal> > HMM::generate_epsilon(std::vector<int>& obs_seq, matrix<mpfr::mpreal>& alpha, matrix<mpfr::mpreal>& beta, std::vector<mpfr::mpreal>& scale_factors) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	std::vector<matrix<mpfr::mpreal> > epsilon(t, matrix<mpfr::mpreal>(n, std::vector<mpfr::mpreal>(n)));
	for (int i = 0; i < t-1; i++) {
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

	std::pair<matrix<mpfr::mpreal>, std::vector<mpfr::mpreal> > alpha_scale_pair = this->generate_forwards(obs_seq);
	matrix<mpfr::mpreal> alpha = alpha_scale_pair.first;
	std::vector<mpfr::mpreal> scale_factors = alpha_scale_pair.second;
	matrix<mpfr::mpreal> beta = this->generate_backwards(obs_seq, scale_factors);
	std::vector<matrix<mpfr::mpreal> > epsilon = this->generate_epsilon(obs_seq, alpha, beta, scale_factors);
	matrix<mpfr::mpreal> gamma = this->generate_gamma(obs_seq, epsilon, scale_factors);

	for (int i = 0; i < t; i++) {
		std::cout << scale_factors[i] << " ";
	}
	std::cout << '\n';

	// re-estimate initial state probabilities
	mpfr::mpreal gammasum = 0;
	for (int i = 0; i < n; i++) {
		gammasum += gamma[0][i];
	}
	for (int i = 0; i < n; i++) {
		this->init_prob[i] = gamma[0][i] / gammasum;
	}
	// re-estimate transition matrix probabilities
	for (int i = 0; i < n; i++) {
		mpfr::mpreal den = 0;
		for (int j = 0; j < t-1; j++) {
			den += gamma[j][i];
		}
		for (int j = 0; j < n; j++) {
			mpfr::mpreal num = 0;
			for (int l = 0; l < t-1; l++) {
				num += epsilon[l][i][j];
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
		mpfr::mpreal den = 0;
		for (int j = 0; j < t; j++) {
			den += gamma[j][i];
		}
		for (int j = 0; j < k; j++) {
			mpfr::mpreal num = 0;
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
	matrix<mpfr::mpreal> a_num(n, std::vector<mpfr::mpreal>(n));
	matrix<mpfr::mpreal> a_den(n, std::vector<mpfr::mpreal>(n));
	matrix<mpfr::mpreal> b_num(n, std::vector<mpfr::mpreal>(k));
	matrix<mpfr::mpreal> b_den(n, std::vector<mpfr::mpreal>(k));
	for (std::vector<int>& obs_seq : obs_seqs) {
		int t = obs_seq.size();
		// initialize auxillary variables
		std::pair<matrix<mpfr::mpreal>, std::vector<mpfr::mpreal> > alpha_scale_pair = this->generate_forwards(obs_seq);
		matrix<mpfr::mpreal> alpha = alpha_scale_pair.first;
		std::vector<mpfr::mpreal> scale_factors = alpha_scale_pair.second;
		matrix<mpfr::mpreal> beta = this->generate_backwards(obs_seq, scale_factors);
		std::vector<matrix<mpfr::mpreal> > epsilon = this->generate_epsilon(obs_seq, alpha, beta, scale_factors);
		matrix<mpfr::mpreal> gamma = this->generate_gamma(obs_seq, epsilon, scale_factors);
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
			mpfr::mpreal den = 0;
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

/**
 * Finds the maximum likelihood state sequence corresponding to the given input sequence
 * phi, the maximum likelihood of being in the current state j at time i, is generated by the recursion
 * 1) if t=0, phi[i][j] = init_prob[j] * emit_prob[j][obs_seq[i]]
 * 2) else, phi[i][j] = max(phi[i-1][prev] * trans_prob[prev][j]) * emit_prob[j][obs_seq[i]]
 * once phi is calculated, backtrack to find the most likely previous state to generate the maximum likelihood state sequence.
 * use logarithms
 *
 * Memory: phi[T][N], prev_state[T][N]
 * Time Complexity: O(T * N^2)
 * 
 * @param obs_seq observation sequence
 * @return maximum likelihood state sequence corresponding to the given input sequence
 */
std::vector<int> HMM::viterbi(std::vector<int>& obs_seq) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	// preprocess logarithms of transition probability matrix
	matrix<mpfr::mpreal> log_trans(n, std::vector<mpfr::mpreal>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			log_trans[i][j] = log(this->trans_prob[i][j]);
		}
	}
	// initialize dynamic programming variables
	matrix<mpfr::mpreal> phi(t, std::vector<mpfr::mpreal>(n));
	matrix<int> prev_state(t, std::vector<int>(n));
	for (int i = 0; i < t; i++) {
		// base case
		if (i == 0) {
			for (int j = 0; j < n; j++) {
				phi[i][j] = log(this->init_prob[j]) + log(this->emit_prob[j][obs_seq[i]]);
				prev_state[i][j] = -1;
			}
		}
		// induction
		else {
			for (int j = 0; j < n; j++) {
				mpfr::mpreal phimax = INT_MIN;
				for (int prev = 0; prev < n; prev++) {
					if (phi[i-1][prev] + log_trans[prev][j] > phimax) {
						phimax = phi[i-1][prev] + log_trans[prev][j];
						prev_state[i][j] = prev;
					}
				}
				phi[i][j] = phimax + log(this->emit_prob[j][obs_seq[i]]);
			}
		}
	}
	// terminate
	std::pair<mpfr::mpreal, int> end_state = {INT_MIN, -1};
	for (int i = 0; i < n; i++) {
		if (phi[t-1][i] > end_state.first) {
			end_state = {phi[t-1][i], prev_state[t-1][i]};
		}
	}
	// backtrack
	int current_state = end_state.second;
	std::vector<int> answer(t);
	for (int i = t-1; i >= 0; i--) {
		answer[i] = current_state;
		current_state = prev_state[i][current_state];
	}
	return answer;
}

void HMM::segment_init(std::vector<int>& obs_seq) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	std::vector<int> state_seq = this->viterbi(obs_seq);
	std::vector<std::vector<mpfr::mpreal> > state_obs_count(n, std::vector<mpfr::mpreal>(k));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			state_obs_count[i][j] = 10;
		}
	}
	for (int i = 0; i < t; i++) {
		state_obs_count[state_seq[i]][obs_seq[i]]++;
	}
	for (int i = 0; i < n; i++) {
		mpfr::mpreal emitsum = 0;
		for (int j = 0; j < k; j++) {
			emitsum += state_obs_count[i][j];
		}
		for (int j = 0; j < k; j++) {
			this->emit_prob[i][j] = state_obs_count[i][j] / emitsum;
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			std::cout << state_obs_count[i][j] << " ";
		}
		std::cout << '\n';
	}
}

void HMM::multi_segment_init(std::vector<std::vector<int> >& obs_seqs) {

}

int main() {
	std::chrono::high_resolution_clock::time_point t0_final = std::chrono::high_resolution_clock::now(); // time before execution

	// set precision
	const int digits = 1000;
	mpfr::mpreal::set_default_prec(mpfr::digits2bits(digits));
	std::cout << "PRECISION: " << mpfr::mpreal::get_default_prec() << "\n";

	int n = 5;
	int k = 50;
	int t = 500;
	matrix<mpfr::mpreal> emit_prob;

	//  Mersene twister, which is a random number generator that has better performance than C++ rand() / srand()
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);

	// generate emission matrix
	emit_prob.resize(n, std::vector<mpfr::mpreal>(k));
	for (int i = 0; i < n; i++) {
		mpfr::mpreal total = 0;
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
	HMM hmm2(emit_prob);

	/*
	// testing single observation sequence
	std::vector<int> obs_seq(t);
	for (int i = 0; i < t; i++) {
		obs_seq[i] = generator() % k;
	}
	for (int iteration = 0; iteration < 30; iteration++) {
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now(); // time before execution
		if (iteration == 0) {
			std::cout << "BEFORE\n";
			// std::cout << hmm << '\n';
			std::cout << hmm.evaluate(obs_seq) << '\n';
			std::cout << "\n------------------\n\n";
		}

		hmm.baum_welch(obs_seq);
		std::cout << "ITERATION " << iteration+1 << '\n';
		// std::cout << hmm << '\n';
		std::cout << hmm.evaluate(obs_seq) << '\n';

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); // time after execution
		std::cout << "EXECUTION TIME " << iteration+1 << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << "\n"; // time elapsed
		std::cout << "\n------------------\n\n";
	}
	*/

	/*
	// testing multiple observation sequences
	std::vector<std::vector<int> > obs_seqs;
	int x = 15;
	obs_seqs.resize(x);
	for (int i = 0; i < x; i++) {
		obs_seqs[i].resize(t);
		for (int j = 0; j < t; j++) {
			obs_seqs[i][j] = generator() % k;
		}	
	}
	for (int iteration = 0; iteration < 3; iteration++) {
		// checking how long each iteration for training and evaluation takes
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now(); // time before execution

		if (iteration == 0) {
			std::cout << "BEFORE" << "\n";
			// std::cout << hmm << '\n';
			mpfr::mpreal tot = 0;
			for (int i = 0; i < x; i++) {
				mpfr::mpreal val = hmm.evaluate(obs_seqs[i]);
				std::cout << val << "\n";
				tot += val;
			}
			std::cout << "total: " << tot << '\n';
			std::cout << "\n------------------\n\n";
		}

		hmm.multi_baum_welch(obs_seqs);
		std::cout << "ITERATION " << iteration+1 << '\n';
		// std::cout << hmm << "\n";
		mpfr::mpreal tot = 0;
		for (int i = 0; i < x; i++) {
			mpfr::mpreal val = hmm.evaluate(obs_seqs[i]);
			std::cout << val << "\n";
			tot += val;
		}
		std::cout << "total: " << tot << '\n';
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); // time after execution
		std::cout << "EXECUTION TIME " << iteration+1 << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << "\n"; // time elapsed
		std::cout << "\n------------------\n\n";
	}
	*/

	/*
	// testing Viterbi's algorithm
	std::vector<int> max_likelihood = hmm.viterbi(obs_seq);
	for (int i = 0; i < t; i++) {
		std::cout << max_likelihood[i] << " ";
		if (i == t-1)
			std::cout << '\n';
	}
	*/

	// testing emission matrix initialization
	std::vector<int> obs_seq(t);
	for (int i = 0; i < t; i++) {
		obs_seq[i] = generator() % k;
	}
	for (int iteration = 0; iteration < 2; iteration++) {
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now(); // time before execution
		if (iteration == 0) {
			std::cout << "BEFORE\n";
			std::cout << hmm << '\n';
			std::cout << hmm.evaluate(obs_seq) << '\n';
			std::cout << "\n------------------\n\n";
		}

		hmm.segment_init(obs_seq);
		std::cout << "ITERATION " << iteration+1 << '\n';
		// std::cout << hmm << '\n';
		std::cout << hmm.evaluate(obs_seq) << '\n';

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); // time after execution
		std::cout << "EXECUTION TIME " << iteration+1 << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << "\n"; // time elapsed
		std::cout << "\n------------------\n\n";
	}
	for (int iteration = 0; iteration < 30; iteration++) {
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now(); // time before execution
		if (iteration == 0) {
			std::cout << "BEFORE\n";
			// std::cout << hmm << '\n';
			std::cout << hmm.evaluate(obs_seq) << '\n';
			std::cout << hmm2.evaluate(obs_seq) << '\n';
			std::cout << "\n------------------\n\n";
		}

		hmm.baum_welch(obs_seq);
		hmm2.baum_welch(obs_seq);
		std::cout << "ITERATION " << iteration+1 << '\n';
		// std::cout << hmm << '\n';
		std::cout << hmm.evaluate(obs_seq) << '\n';
		std::cout << hmm2.evaluate(obs_seq) << '\n';

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); // time after execution
		std::cout << "EXECUTION TIME " << iteration+1 << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << "\n"; // time elapsed
		std::cout << "\n------------------\n\n";
	}
	std::cout << hmm << '\n';
	std::cout << hmm2 << '\n';

	std::chrono::high_resolution_clock::time_point t1_final = std::chrono::high_resolution_clock::now(); // time after execution
	std::cout << "TOTAL EXECUTION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1_final - t0_final).count() << " ms" << "\n"; // time elapsed

	return 0;
}