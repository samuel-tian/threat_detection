#include <utility>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>

#ifndef HMM_H
#define HMM_H

template<typename R, typename S, typename T>
struct triplet {
	R first;
	S second;
	T third;
};

class HMM {
public:

	int N, K; // number of states, number of observations
	std::vector<mpf_class> init_prob; // siez N vector, initial state probabilities
	std::vector<std::vector<mpf_class> > trans_prob; // N x N matrix, probability of transitioning from state i to state j
	std::vector<std::vector<mpf_class> > emit_prob; // N x K matrix, probability of emitting symbol j at state i

	HMM() = default;

	HMM(int num_states, int num_emissions);

	HMM(std::vector<std::vector<mpf_class> >& emit_prob);

	friend std::ostream& operator << (std::ostream& os, const HMM& hmm);

	std::pair<std::vector<std::vector<mpf_class> >, std::vector<mpf_class> > generate_forwards(std::vector<int>& obs_seq);

	std::vector<std::vector<mpf_class> > generate_backwards(std::vector<int>& obs_seq, std::vector<mpf_class>& scale_factors);

	mpf_class evaluate(std::vector<int>& obs_seq);

	std::vector<std::vector<mpf_class> > generate_gamma(std::vector<int>& obs_seq, std::vector<std::vector<mpf_class> >& alpha, std::vector<std::vector<mpf_class> >& beta, std::vector<mpf_class>& scale_factors);

	std::vector<std::vector<std::vector<mpf_class> > > generate_epsilon(std::vector<int>& obs_seq, std::vector<std::vector<mpf_class> >& alpha, std::vector<std::vector<mpf_class> >& beta, std::vector<mpf_class>& scale_factors);

	void baum_welch(std::vector<int>& obs_seq);

	void multi_baum_welch(std::vector<std::vector<int> >& obs_seqs);

};

#endif