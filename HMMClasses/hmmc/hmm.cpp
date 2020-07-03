#include <iostream>
#include <gmp.h>
#include <gmpxx.h>

#include "hmm.h"

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
HMM::HMM(std::vector<std::vector<mpf_class> >& emit_prob) {
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
	this->trans_prob.resize(n, std::vector<mpf_class>);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			this->trans_prob[i][j] = x;
		}
	}
	// initialize emission probabilities to specified matrix
	this->emit_prob = emit_prob;
}

/**
 * Generate scaled forward variable, as specified in Rabiner's paper
 * Unscaled forward variable is defined by the recursion:
 * 1) if t=1, a(t)(n) = init_prob[n] * emit_prob[n][first element in sequence]
 * 2) else, a(t)(n) = emit_prob[n][element t in sequence] * sum_previous{trans_prob[previous][n] * a(t-1)(previous)}
 * Scaled forward variable rescales all values at time t to sum to 1
 * a'(t)(n) = a(t)(n) / sum_i{a(t)(i)}
 *
 * @param obs_seq observation sequence that will be used to generate the forwards recursion
 * @return pair of vectors, containing
 * 1) the scaled forward variables, a matrix of dimension scaled[T][N]
 * 2) the scale factors, a vector of dimension scale_factors[T]
 */
std::pair<std::vector<std::vector<mpf_class> >, std::vector<mpf_class> > HMM::generate_forwards(std::vector<int>& obs_seq) {
	int n = this->N;
	int k = this->K;
	int t = obs_seq.size();

	
}

std::vector<std::vector<mpf_class> > HMM::generate_backwards(std::vector<int>& obs_seq, std::vector<mpf_class> scale_factors) {

}

mpf_class HMM::evaluate(std::vector<int>& obs_seq) {

}

std::vector<std::vector<mpf_class> > HMM::generate_gamma(std::vector<int>& obs_seq, std::vector<mpf_class> alpha, std::vector<mpf_class> beta) {

}

std::vector<std::vector<std::vector<mpf_class> > > HMM::generate_epsilon(std::vector<int>& obs_seq, std::vector<mpf_class> alpha, std::vector<mpf_class> beta) {

}

void HMM::train(std::vector<int> obs_seq) {

}

int main() {

	return 0;
}