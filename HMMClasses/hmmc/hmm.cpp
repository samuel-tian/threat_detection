#include <iostream>
#include <gmp.h>
#include <gmpxx.h>

// #include "hmm.h"

HMM::HMM(int num_states, int num_coordinates) {
	this->N = num_states;
	this->K = num_coordinates * num_coordinates;
	int n = this->N;
	int k = this->K;
	mpf_class x = n;
	x = 1 / x;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			this->trans_prob[i][j] = x;
		}
	}	
}

HMM::HMM(std::vector<std::vector<mpf_class> >& emit_prob) {

}

std::pair<std::vector<int>, std::vector<int> > HMM::generate_forwards(std::vector<int>& obs_seq) {

}

std::vector<int> HMM::generate_backwards(std::vector<int>& obs_seq, std::vector<mpf_class> scale_factors) {

}

void HMM::evaluate(std::vector<int>& obs_seq, mpf_class return_val) {

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