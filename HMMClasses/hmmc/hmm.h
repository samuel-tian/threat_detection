#include <utility>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>
#include "mpreal.h"

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
	std::vector<mpfr::mpreal> init_prob; // siez N vector, initial state probabilities
	std::vector<std::vector<mpfr::mpreal> > trans_prob; // N x N matrix, probability of transitioning from state i to state j
	std::vector<std::vector<mpfr::mpreal> > emit_prob; // N x K matrix, probability of emitting symbol j at state i

	HMM() = default;

	HMM(int num_states, int num_emissions);

	HMM(std::vector<std::vector<mpfr::mpreal> >& emit_prob);

	friend std::ostream& operator << (std::ostream& os, const HMM& hmm);

	std::pair<std::vector<std::vector<mpfr::mpreal> >, std::vector<mpfr::mpreal> > generate_forwards(std::vector<int>& obs_seq);

	std::vector<std::vector<mpfr::mpreal> > generate_backwards(std::vector<int>& obs_seq, std::vector<mpfr::mpreal>& scale_factors);

	mpfr::mpreal evaluate(std::vector<int>& obs_seq);

	std::vector<std::vector<mpfr::mpreal> > generate_gamma(std::vector<int>& obs_seq, std::vector<std::vector<std::vector<mpfr::mpreal> > > epsilon, std::vector<mpfr::mpreal>& scale_factors);

	std::vector<std::vector<std::vector<mpfr::mpreal> > > generate_epsilon(std::vector<int>& obs_seq, std::vector<std::vector<mpfr::mpreal> >& alpha, std::vector<std::vector<mpfr::mpreal> >& beta, std::vector<mpfr::mpreal>& scale_factors);

	void baum_welch(std::vector<int>& obs_seq);

	void multi_baum_welch(std::vector<std::vector<int> >& obs_seqs);

};

#endif