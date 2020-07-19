#include "kmeans.h"

#include <iostream>
#include <gmp.h>
#include <mpfr.h>
#include "mpreal.h"
#include <vector>
#include <chrono>
#include <random>

namespace kmeans {

	mpfr::mpreal epsilon;

	/**
	 * Set threshold for distortion in termination condition
	 * 
	 * @param epsilon new threshold for distortion
	 */
	void set_threshold(mpfr::mpreal e) {
		epsilon = e;
	}

	/**
	 * Get threshold for distortion in termination condition
	 * 
	 * @return threshold for distortion
	 */
	mpfr::mpreal get_threshold() {
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
	mpfr::mpreal compute_distortion(std::vector<param<mpfr::mpreal> > & centroids, std::vector<std::vector<param<mpfr::mpreal> > >& clustered_paths) {
		int n = 0;
		int k = centroids.size();
		mpfr::mpreal mse = 0;
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < clustered_paths[i].size(); j++) {
				mpfr::mpreal dist = difference(centroids[i], clustered_paths[i][j]);
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
	std::vector<param<mpfr::mpreal> > initialize_centroids(std::vector<param<mpfr::mpreal> >& paths, int k) {
		std::vector<param<mpfr::mpreal> > centroids(k);
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
	std::vector<std::vector<param<mpfr::mpreal> > > recluster(std::vector<param<mpfr::mpreal> >& centroids, std::vector<param<mpfr::mpreal> >& paths) {
		int k = centroids.size();
		int n = paths.size();
		std::vector<std::vector<param<mpfr::mpreal> > > clustered_paths;
		clustered_paths.resize(k);
		for (int i = 0; i < n; i++) {
			mpfr::mpreal min_dist = 1<<30;
			int ind = 0;
			for (int j = 0; j < k; j++) {
				mpfr::mpreal dist = difference(paths[i], centroids[j]);
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
	void update_clusters(std::vector<param<mpfr::mpreal> >& centroids, std::vector<std::vector<param<mpfr::mpreal> > >& clustered_paths) {
		int k = centroids.size();
		int x = centroids[0].size();
		for (int i = 0; i < k; i++) {
			param<mpfr::mpreal> p(x);
			for (int j = 0; j < clustered_paths[i].size(); j++) {
				for (int l = 0; l < x; l++) {
					mpfr::mpreal v = p.get(l) + clustered_paths[i][j].get(l);
					p.set(l, v);
				}
			}
			mpfr::mpreal sz = clustered_paths[i].size();
			for (int j = 0; j < x; j++) {
				mpfr::mpreal v = p.get(j) / sz;
				p.set(j, v);
				centroids[i] = p;
			}
		}
	}

	/**
	 * Clusters given path into one of the k centroids
	 * @param centroids current set of centroids
	 * @param path given path that needs to be clustered
	 * @return index of the nearest centroid
	 */
	int nearest_centroid(std::vector<param<mpfr::mpreal> >& centroids, param<mpfr::mpreal>& path) {
		int k = centroids.size();
		mpfr::mpreal min_dist = 1<<30;
		int ind = 0;
		for (int i = 0; i < k; i++) {
			mpfr::mpreal dist = difference(centroids[i], path);
			if (dist < min_dist) {
				min_dist = dist;
				ind = i;
			}
		}
		return ind;
	}

	std::vector<param<mpfr::mpreal> > iterative_LBG(std::vector<param<mpfr::mpreal> >& paths, int k) {
		int n = paths.size();
		std::vector<param<mpfr::mpreal> > centroids = initialize_centroids(paths, k);
		std::vector<std::vector<param<mpfr::mpreal> > > clustered_paths = recluster(centroids, paths);
		mpfr::mpreal prev_distortion = 1<<30;
		int iteration_num = 1;
		while (true) {
			clustered_paths = recluster(centroids, paths);
			mpfr::mpreal cur_distortion = compute_distortion(centroids, clustered_paths);
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

/*
int main() {
	using namespace kmeans;

	const int digits = 1000;
	mpfr::mpreal::set_default_prec(mpfr::digits2bits(digits));
	std::cout << "PRECISION: " << mpfr::mpreal::get_default_prec() << '\n';

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);

	mpfr::mpreal epsilon = 0.0001;
	set_threshold(epsilon);

	int x = 3;
	int k = 4;
	int n = 50;
	std::vector<param<mpfr::mpreal> > paths(n, param<mpfr::mpreal>(x));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < x; j++) {
			mpfr::mpreal v = generator()%15;
			paths[i].set(j, v);
		}
	}
	std::cout << "PATHS: ";
	for (int i = 0; i < n; i++) {
		std::cout << paths[i] << " ";
		if (i == n-1)
			std::cout << '\n';
	}
	std::vector<param<mpfr::mpreal> > centroids = iterative_LBG(paths, k);
	std::cout << "CENTROIDS: ";
	for (int i = 0; i < k; i++) {
		std::cout << centroids[i] << '\n';
		if (i == k-1)
			std::cout << '\n';
	}
	return 0;
}
*/