#include <stdio.h>
#include <utility>
#include <vector>
#include <boost/multiprecision/mpfr.hpp>
// #include <gmp.h>
// #include <gmpxx.h>
// #include <mpfr.h>
// #include "mpreal.h"

#ifndef HMM_H
#define HMM_H

class HMM {
public:

	int N, K; // number of states, number of observations
	std::vector<boost::multiprecision::mpfr_float> init_prob; // siez N vector, initial state probabilities
	std::vector<std::vector<boost::multiprecision::mpfr_float> > trans_prob; // N x N matrix, probability of transitioning from state i to state j
	std::vector<std::vector<boost::multiprecision::mpfr_float> > emit_prob; // N x K matrix, probability of emitting symbol j at state i

	HMM() = default;

	HMM(int num_states, int num_emissions);

	HMM(std::vector<std::vector<boost::multiprecision::mpfr_float> >& emit_prob);

	HMM(int N, int K, std::vector<boost::multiprecision::mpfr_float>& init_prob, std::vector<std::vector<boost::multiprecision::mpfr_float> >& trans_prob, std::vector<std::vector<boost::multiprecision::mpfr_float> >& emit_prob);

	friend std::ostream& operator << (std::ostream& os, const HMM& hmm);

	std::pair<std::vector<std::vector<boost::multiprecision::mpfr_float> >, std::vector<boost::multiprecision::mpfr_float> > generate_forwards(std::vector<int>& obs_seq);

	std::vector<std::vector<boost::multiprecision::mpfr_float> > generate_backwards(std::vector<int>& obs_seq, std::vector<boost::multiprecision::mpfr_float>& scale_factors);

	boost::multiprecision::mpfr_float evaluate(std::vector<int>& obs_seq);

	std::vector<std::vector<boost::multiprecision::mpfr_float> > generate_gamma(std::vector<int>& obs_seq, std::vector<std::vector<std::vector<boost::multiprecision::mpfr_float> > > epsilon, std::vector<boost::multiprecision::mpfr_float>& scale_factors);

	std::vector<std::vector<std::vector<boost::multiprecision::mpfr_float> > > generate_epsilon(std::vector<int>& obs_seq, std::vector<std::vector<boost::multiprecision::mpfr_float> >& alpha, std::vector<std::vector<boost::multiprecision::mpfr_float> >& beta, std::vector<boost::multiprecision::mpfr_float>& scale_factors);

	void baum_welch(std::vector<int>& obs_seq);

	void multi_baum_welch(std::vector<std::vector<int> >& obs_seqs);

	std::vector<int> viterbi(std::vector<int>& obs_seq);

	void segment_init(std::vector<int>& obs_seq);

	void multi_segment_init(std::vector<std::vector<int> >& obs_seqs);

};

#endif // HMM_H