#include <iostream>
#include <stdio.h>
#include <fstream>
#include <gmp.h>
#include <mpfr.h>
#include <string>
#include <algorithm>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "./hmm/hmm.h"
#include "./kmeans/kmeans.h"

using namespace kmeans;

template<typename T>
using matrix = std::vector<std::vector<T> >;

bool is_empty(std::ifstream& fin) {
	return fin.peek() == std::ifstream::traits_type::eof();
}

std::pair<std::vector<param<mpfr_float> >, std::vector<traj<mpfr_float> > > read_params_raw(std::string s) {
	std::ifstream in("data/raw/" + s + "_seg.dat");
	using namespace kmeans;
	std::vector<param<mpfr_float> > paths; // contains all parameterizations
	std::vector<traj<mpfr_float> > trajs; // contains all trajectories
	int num_traj; // number of trajectories
	int t, k; // number of observations per trajectory, number of coordinates per observation

	std::string in_buffer;
	std::vector<std::string> buffer_vector;

	// parse data parameters
	std::getline(in, in_buffer);
	boost::split(buffer_vector, in_buffer, boost::is_any_of("\t "));
	num_traj = boost::lexical_cast<int>(buffer_vector[0]);
	t = boost::lexical_cast<int>(buffer_vector[1]);
	k = boost::lexical_cast<int>(buffer_vector[2]);

	int cnt = 1;
	for (int i = 0; i < num_traj; i++) {
		traj<mpfr_float> traj(t);
		for (int j = 0; j < t; j++) {
			param<mpfr_float> path(k);
			for (int l = 0; l < k; l++) {
				std::getline(in, in_buffer);
				mpfr_float v = boost::lexical_cast<mpfr_float>(in_buffer);
				// std::cerr << cnt++ << " " << v << '\n';
				path.set(l, v);
			}
			paths.push_back(path);
			traj.set(j, path);
		}
		trajs.push_back(traj);
	}
	in.close();
	return std::make_pair(paths, trajs);
}

std::vector<param<mpfr_float> > cluster_params(std::string s, std::vector<param<mpfr_float> >& paths, int k) {
	std::cerr << "clustering " << s << '\n';
	std::ifstream fin("data/centroids/" + s + "_seg_centroids.dat");
	std::vector<param<mpfr_float> > centroids;
	if (is_empty(fin)) { // file is empty, centroids have not been generated
		std::cerr << "generating new centroids" << '\n';
		fin.close();
		std::ofstream fout("data/centroids/" + s + "_seg_centroids.dat");
		std::vector<param<mpfr_float> > centroids = iterative_LBG(paths, k);
		for (int i = 0; i < centroids.size(); i++) {
			fout << centroids[i] << '\n';
		}
		fout.close();
	}
	else {
		std::cerr << "reading centroids from file" << '\n';
		fin.seekg(0, fin.beg);
		std::string in_buffer;
		std::vector<std::string> buffer_vector;
		while (std::getline(fin, in_buffer)) {
			if (in_buffer.empty()) break;
			in_buffer = in_buffer.substr(1, in_buffer.size()-2);
			boost::split(buffer_vector, in_buffer, boost::is_any_of("\t, "));
			param<mpfr_float> p(buffer_vector.size());
			for (int i = 0; i < buffer_vector.size(); i++) {
				if (buffer_vector[i].empty()) continue;
				mpfr_float v = boost::lexical_cast<mpfr_float>(buffer_vector[i]);
				p.set(i, v);
			}
			centroids.push_back(p);
		}
		fin.close();
	}
	return centroids;
}

std::vector<std::vector<int> > cluster_traj(std::vector<traj<mpfr_float> >& trajectories, std::vector<param<mpfr_float> >& centroids) {
	int n = trajectories.size();
	std::vector<std::vector<int> > traj_mapped(n);
	for (int i = 0; i < n; i++) {
		traj_mapped[i].resize(trajectories[i].size());
		for (int j = 0; j < traj_mapped[i].size(); j++) {
			param<mpfr_float> tmp = trajectories[i].get(j);
			traj_mapped[i][j] = nearest_centroid(centroids, tmp);
		}
	}
	return traj_mapped;
}

HMM train_HMM(std::string s, std::vector<std::vector<int> >& trajs_mapped, int n, int k) {
	std::cerr << "training " << s << '\n';
	std::ifstream fin("data/hmm/" + s + "_seg_HMM.dat");
	HMM hmm;
	if (is_empty(fin)) {
		std::cerr << "generating new HMM" << '\n';
		fin.close();
		std::ofstream out("data/hmm/" + s + "_seg_HMM.dat");
		HMM hmm(n, k);
		// initialize HMM using segmental kmeans segmentation
		for (int iteration = 0; iteration < 10; iteration++) {
			std::cerr << "INITIALIZE ITERATION " << iteration << '\n';
			std::cerr << "----------------------\n";
			std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

			hmm.multi_segment_init(trajs_mapped);

			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			std::cerr << "EXECUTION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms\n\n";
		}
		for (int iteration = 0; iteration < 10; iteration++) {
			std::cerr << "REESTIMATION ITERATION " << iteration << '\n';
			std::cerr << "----------------------\n";
			std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

			hmm.multi_baum_welch(trajs_mapped);
			
			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			std::cerr << "EXECUTION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms\n\n";
		}
		out << hmm << '\n';
		out.close();
	}
	else {
		std::cerr << "reading HMM from file" << '\n';
		fin.seekg(0, fin.beg);
		std::string in_buffer;
		std::vector<std::string> buffer_vector;
		// n
		std::getline(fin, in_buffer);
		int n = boost::lexical_cast<int>(in_buffer);
		// k
		std::getline(fin, in_buffer);
		int k = boost::lexical_cast<int>(in_buffer);
		// initial state probabilities
		std::vector<mpfr_float> init_prob(n);
		std::getline(fin, in_buffer);
		buffer_vector = boost::split(buffer_vector, in_buffer, boost::is_any_of("\t "));
		for (int i = 0; i < n; i++) {
			init_prob[i] = boost::lexical_cast<mpfr_float>(buffer_vector[i]);
		}
		// transition probabilities
		std::vector<std::vector<mpfr_float> > trans_prob(n, std::vector<mpfr_float>(n));
		for (int i = 0; i < n; i++) {
			std::getline(fin, in_buffer);
			buffer_vector = boost::split(buffer_vector, in_buffer, boost::is_any_of("\t "));
			for (int j = 0; j < n; j++) {
				trans_prob[i][j] = boost::lexical_cast<mpfr_float>(buffer_vector[j]);
			}
		}
		// emission probabilities
		std::vector<std::vector<mpfr_float> > emit_prob(n, std::vector<mpfr_float>(k));
		for (int i = 0; i < n; i++) {
			std::getline(fin, in_buffer);
			buffer_vector = boost::split(buffer_vector, in_buffer, boost::is_any_of("\t "));
			for (int j = 0; j < k; j++) {
				emit_prob[i][j] = boost::lexical_cast<mpfr_float>(buffer_vector[j]);
			}
		}
		hmm = HMM(n, k, init_prob, trans_prob, emit_prob);
	}
	return hmm;
}

void bucketize(std::string s, std::vector<traj<mpfr_float> > trajs, std::vector<param<mpfr_float> > centroids, HMM hmm, int num_buckets) {
	std::cerr << "bucketizing " << s << '\n';
	std::ofstream fout("data/bucketized/" + s + "_seg_bucketized.dat");
	std::vector<std::pair<mpfr_float, int> > log_likelihoods;
	std::vector<std::vector<int> > trajs_mapped = cluster_traj(trajs, centroids);
	for (int i = 0; i < trajs_mapped.size(); i++) {
		mpfr_float prob = hmm.evaluate(trajs_mapped[i]);
		log_likelihoods.emplace_back(prob, i);
	}
	std::sort(log_likelihoods.begin(), log_likelihoods.end());
	mpfr_float min_val = log_likelihoods[0].first;
	mpfr_float bucket_size = (log_likelihoods[log_likelihoods.size()-1].first - log_likelihoods[0].first) / num_buckets;
	std::vector<std::vector<int> > buckets(num_buckets);
	for (int i = 0; i < log_likelihoods.size(); i++) {
		int bucket_num = (int) ((log_likelihoods[i].first - min_val) / bucket_size);
		if (bucket_num >= num_buckets)
			bucket_num = num_buckets - 1;
		buckets[bucket_num].push_back(log_likelihoods[i].second);
	}
	fout << num_buckets << '\n';
	for (int i = 0; i < num_buckets; i++) {
		fout << buckets[i].size() << '\n';
		for (int j = 0; j < buckets[i].size(); j++) {
			fout << trajs[buckets[i][j]] << '\n';
		}
	}
}

void preprocess(std::string s, int n = 10, int k = 256) {
	std::pair<std::vector<param<mpfr_float> >, std::vector<traj<mpfr_float> > > tmp = read_params_raw(s);
	std::vector<param<mpfr_float> > paths = tmp.first;
	std::vector<traj<mpfr_float> > trajs = tmp.second;
	std::vector<param<mpfr_float> > centroids = cluster_params(s, paths, k);
	std::vector<std::vector<int> > trajs_mapped = cluster_traj(trajs, centroids);
	for (int i = 0; i < trajs_mapped.size(); i++) {
		for (int j = 0; j < trajs_mapped[i].size(); j++) {
			std::cerr << trajs_mapped[i][j] << " ";
			if (j == trajs_mapped[i].size()-1)
				std::cerr << '\n';
		}
	}
	HMM hmm = train_HMM(s, trajs_mapped, n, k);
	bucketize(s, trajs, centroids, hmm, 20);
}

int main() {
	set_threshold(0.01);
	mpfr_float::default_precision(250);

	preprocess("circling");
}