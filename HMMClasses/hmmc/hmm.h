#include <utility>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>

#ifndef HMM_H
#define HMM_H

class HMM {
public:

	int N, K; // number of states, number of observations
	std::vector<mpf_class> init_prob; // siez N vector, initial state probabilities
	std::vector<std::vector<mpf_class> > trans_prob; // N x N matrix, probability of transitioning from state i to state j
	std::vector<std::vector<mpf_class> > emit_prob; // N x K matrix, probability of emitting symbol j at state i

	HMM() = default;

	HMM(int num_states, int num_coordinates);

	HMM(std::vector<std::vector<mpf_class> >& emit_prob);

	std::pair<std::vector<int>, std::vector<int> > generate_forwards(std::vector<int>& obs_seq);

	std::vector<int> generate_backwards(std::vector<int>& obs_seq, std::vector<mpf_class> scale_factors);

	void evaluate(std::vector<int>& obs_seq, mpf_class return_val);

	std::vector<std::vector<mpf_class> > generate_gamma(std::vector<int>& obs_seq, std::vector<mpf_class> alpha, std::vector<mpf_class> beta);

	std::vector<std::vector<std::vector<mpf_class> > > generate_epsilon(std::vector<int>& obs_seq, std::vector<mpf_class> alpha, std::vector<mpf_class> beta);

	void train(std::vector<int> obs_seq);

};

#endif