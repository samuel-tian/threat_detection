#include <iostream>
#include <stdio.h>
#include <fstream>
#include <gmp.h>
#include <mpfr.h>
#include <string>

#include "./hmm/hmm.h"
#include "./kmeans/kmeans.h"

template<typename T>
using matrix = std::vector<std::vector<T> >;

void read(std::string s) {
	std::ifstream in("data/" + s + ".dat");
	std::ofstream out("data/" + s + "HMM.dat");
	using namespace kmeans;
	std::vector<param<mpfr::mpreal> > paths; // contains all parameterizations
	std::vector<traj<mpfr::mpreal> > trajs; // contains all trajectories
	int num_traj; // number of trajectories
	in >> num_traj;
	paths.resize(num_traj);
	for (int i = 0; i < num_traj; i++) {
		int t, k; // number of observations per trajectory, number of coordinates per observation
		in >> t >> k;
		traj<mpfr::mpreal> traj(t);
		for (int j = 0; j < t; j++) {
			param<mpfr::mpreal> path(k);
			for (int l = 0; l < k; l++) {
				mpfr::mpreal v;
				in >> v;
				path.set(l, v);
			}
			paths.push_back(path);
			traj.set(j, path);
		}
		trajs.push_back(traj);
	}
	int k = 256; // number of clusters / observations
	std::vector<param<mpfr::mpreal> > centroids = iterative_LBG(paths, k);
	std::vector<std::vector<int> > trajs_mapped(num_traj);
	for (int i = 0; i < num_traj; i++) {
		trajs_mapped[i].resize(trajs[i].size());
		for (int j = 0; j < trajs[i].size(); j++) {
			param<mpfr::mpreal> tmp_param = trajs[i].get(j);
			trajs_mapped[i][j] = nearest_centroid(centroids, tmp_param);
		}
	}
	int n = 10; // number of states in the HMM
	HMM hmm(n, k);
	for (int iteration = 0; iteration < 10; iteration++) {
		std::cout << "INITIALIZE ITERATION " << iteration << '\n';
		std::cout << "----------------------\n";
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

		hmm.multi_segment_init(trajs_mapped);

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		std::cout << "EXECUTION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms\n\n";
	}
	for (int iteration = 0; iteration < 30; iteration++) {
		std::cout << "REESTIMATION ITERATION " << iteration << '\n';
		std::cout << "----------------------\n";
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

		hmm.multi_baum_welch(trajs_mapped);
		
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		std::cout << "EXECUTION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms\n";
	}
	out << hmm << '\n';
}

int main() {
	kmeans::set_threshold(0.0001);

	// read("normal");
	// read("following");	
}