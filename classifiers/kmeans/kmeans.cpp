#include "kmeans.h"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <gmp.h>
#include <mpfr.h>
#include "mpreal.h"
#include <vector>
#include <chrono>
#include <random>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/multiprecision/mpfr.hpp>

using namespace boost::multiprecision;

namespace kmeans {

	/**
	 * Measure of distortion between two parameterizations
	 * Returns the Euclidean distance between the centroids of the two parameterizations
	 *
	 * @param a first parameterization
	 * @param b second parameterization
	 * @return Euclidean distance between parameterizations a and b
	 */
	template<typename T>
	T difference(param<T>& a, param<T>& b) {
		std::pair<T, T> centroid_a = a.get_centroid();
		std::pair<T, T> centroid_b = b.get_centroid();
		T ret = 0;
		ret += (centroid_a.first - centroid_b.first) * (centroid_a.first - centroid_b.first);
		ret += (centroid_a.second - centroid_b.second) * (centroid_a.second - centroid_b.second);
		return ret;
	}

	mpfr_float epsilon;

	/**
	 * Set threshold for distortion in termination condition
	 * 
	 * @param epsilon new threshold for distortion
	 */
	void set_threshold(mpfr_float e) {
		epsilon = e;
	}

	/**
	 * Get threshold for distortion in termination condition
	 * 
	 * @return threshold for distortion
	 */
	mpfr_float get_threshold() {
		return kmeans::epsilon;
	}

	/**
	 * Computes the distortion of representing a set of parameterizations with the specified centroids
	 * Uses MSE as a distortion measure, which is 1/n * sum {error^2}
	 * 
	 * @param centroids initial set of centroids
	 * @param clustered_paths set of trajectories clustered into groups based on their closest centroid
	 * @return mean squared error of cluster
	 */
	mpfr_float compute_distortion(std::vector<param<mpfr_float> > & centroids, std::vector<std::vector<param<mpfr_float> > >& clustered_paths) {
		int n = 0;
		int k = centroids.size();
		mpfr_float mse = 0;
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < clustered_paths[i].size(); j++) {
				mpfr_float dist = difference(centroids[i], clustered_paths[i][j]);
				mse += dist * dist;
				n++;
			}
		}
		return mse / n;
	}

	/**
	 * Initialize the set of centroids by choosing the first k vectors in the set of parameterizations
	 *
	 * @param paths set of all parameterizations
	 * @param k desired number of centroids
	 * @return set of centroids of length k
	 */
	std::vector<param<mpfr_float> > initialize_centroids(std::vector<param<mpfr_float> >& paths, int k) {
		std::vector<param<mpfr_float> > centroids(k);
		for (int i = 0; i < k; i++) {
			centroids[i] = paths[i];
		}
		return centroids;
	}

	/**
	 * Group each trajectory with its nearest centroid
	 *
	 * @param centroids set of current centroids
	 * @param paths set of all paths
	 * @return reclustered paths
	 */
	std::vector<std::vector<param<mpfr_float> > > recluster(std::vector<param<mpfr_float> >& centroids, std::vector<param<mpfr_float> >& paths) {
		int k = centroids.size();
		int n = paths.size();
		std::vector<std::vector<param<mpfr_float> > > clustered_paths;
		clustered_paths.resize(k);
		for (int i = 0; i < n; i++) {
			mpfr_float min_dist = 1<<30;
			int ind = 0;
			for (int j = 0; j < k; j++) {
				mpfr_float dist = difference(paths[i], centroids[j]);
				if (dist < min_dist) {
					min_dist = dist;
					ind = j;
				}
			}
			clustered_paths[ind].push_back(paths[i]);
		}
		return clustered_paths;
	}

	/**
	 * Update centroid to the centroid of the vectors clustered around it
	 * @param centroids current set of centroids
	 * @param clustered_paths set of all paths clustered to nearest centroid
	 */
	void update_clusters(std::vector<param<mpfr_float> >& centroids, std::vector<std::vector<param<mpfr_float> > >& clustered_paths) {
		int k = centroids.size();
		int x = centroids[0].size();
		for (int i = 0; i < k; i++) {
			param<mpfr_float> p(x);
			for (int j = 0; j < clustered_paths[i].size(); j++) {
				for (int l = 0; l < x; l++) {
					mpfr_float v = p.get(l) + clustered_paths[i][j].get(l);
					p.set(l, v);
				}
			}
			mpfr_float sz = clustered_paths[i].size();
			for (int j = 0; j < x; j++) {
				mpfr_float v = p.get(j) / sz;
				p.set(j, v);
			}
			centroids[i] = p;
		}
	}

	/**
	 * Clusters given path into one of the k centroids
	 * @param centroids current set of centroids
	 * @param path given path that needs to be clustered
	 * @return index of the nearest centroid
	 */
	int nearest_centroid(std::vector<param<mpfr_float> >& centroids, param<mpfr_float>& path) {
		int k = centroids.size();
		mpfr_float min_dist = 1<<30;
		int ind = 0;
		for (int i = 0; i < k; i++) {
			mpfr_float dist = difference(centroids[i], path);
			if (dist < min_dist) {
				min_dist = dist;
				ind = i;
			}
		}
		return ind;
	}

	std::vector<param<mpfr_float> > iterative_LBG(std::vector<param<mpfr_float> >& paths, int k) {
		int n = paths.size();
		std::vector<param<mpfr_float> > centroids = initialize_centroids(paths, k);
		std::cerr << "initialize" << '\n';
		std::vector<std::vector<param<mpfr_float> > > clustered_paths = recluster(centroids, paths);
		std::cerr << "recluster" << '\n';
		mpfr_float prev_distortion = 1<<30;
		int iteration_num = 1;
		while (true) {
			clustered_paths = recluster(centroids, paths);
			mpfr_float cur_distortion = compute_distortion(centroids, clustered_paths);
			std::cerr << "iteration " << iteration_num << ": " << cur_distortion << '\n';
			if (abs(cur_distortion - prev_distortion) < get_threshold()) {
				break;
			}
			prev_distortion = cur_distortion;
			update_clusters(centroids, clustered_paths);
			iteration_num++;
		}
		return centroids;
	}

}

/*int main() {
	using namespace kmeans;

	std::ofstream out("clustered.dat", std::ofstream::out);

	const int digits = 1000;
	mpfr_float::set_default_prec(mpfr::digits2bits(digits));
	std::cout << "PRECISION: " << mpfr_float::get_default_prec() << '\n';

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);

	mpfr_float epsilon = 0.0001;
	set_threshold(epsilon);

	int x = 2;
	int k = 10;
	int n = 500;
	std::vector<param<mpfr_float> > paths(n, param<mpfr_float>(x));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < x; j++) {
			mpfr_float v = 2.5 + 15 * (generator()%1000000) / 1000000.0;
			paths[i].set(j, v);
		}
	}
	std::vector<param<mpfr_float> > centroids = iterative_LBG(paths, k);
	std::vector<std::vector<param<mpfr_float> > > clustered_paths = recluster(centroids, paths);
	out << k << '\n';
	for (int i = 0; i < k; i++) {
		out << centroids[i] << '\n';
		out << clustered_paths[i].size() << '\n';
		for (int j = 0; j < clustered_paths[i].size(); j++) {
			out << clustered_paths[i][j] << '\n';
		}
	}
	return 0;
}*/

/*int main() {
	mpfr_float::default_precision(250);

	std::string s = "circling";
	std::ifstream in("../data/" + s + "_seg.dat");
	std::ofstream out("../data/" + s + "_seg_HMM.dat");

	using namespace kmeans;
	set_threshold(0.01);
	std::vector<param<mpfr_float> > paths; // contains all parameterizations
	std::vector<traj<mpfr_float> > trajs; // contains all trajectories
	int num_traj; // number of trajectories
	int t, k; // number of observations per trajectory, number of coordinates per observation

	// parse string of file parameters
	std::string params_str;
	std::getline(in, params_str);
	std::vector<std::string> params;
	boost::split(params, params_str, boost::is_any_of("\t "));
	num_traj = boost::lexical_cast<int>(params[0]);
	t = boost::lexical_cast<int>(params[1]);
	k = boost::lexical_cast<int>(params[2]);

	int cnt = 0;
	for (int i = 0; i < num_traj; i++) {
		traj<mpfr_float> traj(t);
		for (int j = 0; j < t; j++) {
			param<mpfr_float> path(k);
			for (int l = 0; l < k; l++) {
				std::string parameter_str;
				std::getline(in, parameter_str);
				mpfr_float v = boost::lexical_cast<mpfr_float>(parameter_str);
				// std::cerr << v << '\n';
				path.set(l, v);
			}
			paths.push_back(path);
			traj.set(j, path);
		}
		trajs.push_back(traj);
	}
	k = 256; // number of clusters / observations
	std::vector<param<mpfr_float> > centroids = iterative_LBG(paths, k);
	std::vector<std::vector<int> > trajs_mapped(num_traj);
	for (int i = 0; i < num_traj; i++) {
		trajs_mapped[i].resize(trajs[i].size());
		for (int j = 0; j < trajs[i].size(); j++) {
			param<mpfr_float> tmp_param = trajs[i].get(j);
			trajs_mapped[i][j] = nearest_centroid(centroids, tmp_param);
		}
	}
}*/