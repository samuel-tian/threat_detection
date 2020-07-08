#include <iostream>
#include <stdio.h>
#include <vector>
#include <mpfr.h>
#include <gmp.h>
#include <climits>
#include <chrono>
#include <random>
#include "mpreal.h"

#include "kmeans.h"

using kmeans::Path;

/**
 * Evalutes Euclidean distance between two vectors, which is defined as sqrt(sum{(v1_i - v2_i)^2})
 * 
 * @param a vector 1
 * @param b vector 2
 * @return Euclidean distance between vector 1 and vector 2
 */
template<typename T>
T kmeans::dif(Path<T>& a, Path<T>& b) {
	T ret = 0;
	for (int i = 0; i < a.size(); i++) {
		ret += (b[i]-a[i]) * (b[i]-a[i]);
	}
	return sqrt(ret);
}

/**
 * Set a threshold for the termination condition.
 *
 * @param epsilon specified level of precision
 * @return void
 */
void kmeans::set_threshold(mpfr::mpreal epsilon) {
	kmeans::epsilon = epsilon;
}

/**
 * Compute the distortion of representing the set of parameterizations with the specified centroids
 * 
 * @param clusters set of k centroids that will indicate the cluster groups
 * @param paths set of all n parameterizations, grouped by closest centroid
 * @return MSE of clustering scheme
 */
mpfr::mpreal kmeans::compute_distortion(std::vector<Path<mpfr::mpreal> >& clusters, std::vector<std::vector<Path<mpfr::mpreal> > >& paths) {
	int n = 0;
	int k = clusters.size();
	mpfr::mpreal mse = 0; 
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < paths[i].size(); j++) {
			mpfr::mpreal dis = kmeans::dif<mpfr::mpreal>(clusters[i], paths[i][j]);
			mse += dis * dis;
			n++;
		}
	}
	return mse;
}

/**
 * Initialize the set of clusters, by choosing the first k vectors in the set of parameterizations
 *
 * @param paths set of all parameterizations
 * @param desired number of clusters, k
 * @return set of clusters of size k
 */
std::vector<Path<mpfr::mpreal> > kmeans::initialize_clusters(std::vector<Path<mpfr::mpreal> >& paths, int k) {
	int x = paths[0].size();
	std::vector<Path<mpfr::mpreal> > clusters(k, Path<mpfr::mpreal>(x));
	for (int i = 0; i < k; i++) {
		clusters[i] = paths[i];
	}
	return clusters;
}

/**
 * Group each trajectory to its nearest centroid, according to Euclidean distance
 * 
 * @param clusters current centroids
 * @param paths set of all paths
 * @return void
 */
std::vector<std::vector<Path<mpfr::mpreal> > > kmeans::recluster(std::vector<Path<mpfr::mpreal> >& clusters, std::vector<Path<mpfr::mpreal> >& paths) {
	int k = clusters.size();
	int n = paths.size();
	std::vector<std::vector<Path<mpfr::mpreal> > > clustered_paths;
	clustered_paths.resize(k);
	for (int i = 0; i < n; i++) {
		mpfr::mpreal min_dist = INT_MAX;
		int ind = 0;
		for (int j = 0; j < k; j++) {
			mpfr::mpreal dis = kmeans::dif<mpfr::mpreal>(paths[i], clusters[j]);
			if (dis < min_dist) {
				min_dist = dis;
				ind = j;
			}
		}
		clustered_paths[ind].push_back(paths[i]);
	}
	return clustered_paths;
}

/**
 * Updates centroids, given the set of vectors that are clustered around it
 * 
 * @param clusters current set of centroids
 * @param clustered_paths set of all vectors, grouped by closest centroid
 */
void kmeans::update_clusters(std::vector<Path<mpfr::mpreal> >& clusters, std::vector<std::vector<Path<mpfr::mpreal> > >& clustered_paths) {
	int k = clusters.size();
	for (int i = 0; i < k; i++) {
		Path<mpfr::mpreal> p(clusters[i].size());
		for (int j = 0; j < clustered_paths[i].size(); j++) {
			p += clustered_paths[i][j];
		}
		mpfr::mpreal sz = (int)clustered_paths[i].size();
		p /= sz;
		clusters[i] = p;
	}
}

std::pair<std::vector<Path<mpfr::mpreal> >, std::vector<std::vector<Path<mpfr::mpreal> > > > kmeans::iterative_LBG(std::vector<Path<mpfr::mpreal> >& paths, int k) {
	int n = paths.size();
	std::cout << "PATHS: ";
	for (int i = 0; i < n; i++) {
		std::cout << paths[i] << " ";
	}
	std::cout << '\n';
	std::vector<Path<mpfr::mpreal> > clusters = kmeans::initialize_clusters(paths, k);
	mpfr::mpreal prev_distortion = INT_MAX;
	int iteration_num = 1;
	while (true) {
		std::cout << "----------------\n";
		std::cout << "ITERATION " << iteration_num << '\n';
		std::cout << "----------------\n";
		std::vector<std::vector<Path<mpfr::mpreal> > > clustered_paths = kmeans::recluster(clusters, paths);
		for (int i = 0; i < k; i++) {
			std::cout << i+1 << " - " << clusters[i] << ": ";
			for (int j = 0; j < clustered_paths[i].size(); j++) {
				std::cout << clustered_paths[i][j] << " ";
			}
			std::cout << '\n';
		}
		mpfr::mpreal cur_distortion = kmeans::compute_distortion(clusters, clustered_paths);
		std::cout << "distortion " << cur_distortion << '\n';
		if (abs(cur_distortion - prev_distortion) / cur_distortion < kmeans::epsilon) {
			break;
		}
		prev_distortion = cur_distortion;
		kmeans::update_clusters(clusters, clustered_paths);
		iteration_num++;
	}

	return std::make_pair(std::vector<Path<mpfr::mpreal> >(), std::vector<std::vector<Path<mpfr::mpreal> > >());
}

int main() {

	// set precision
	const int digits = 1000;
	mpfr::mpreal::set_default_prec(mpfr::digits2bits(digits));
	std::cout << "PRECISION: " << mpfr::mpreal::get_default_prec() << "\n";

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);

	using namespace kmeans;
	mpfr::mpreal epsilon = 0.001;
	set_threshold(epsilon);

	int x = 3;
	int k = 4;
	int n = 50;
	std::vector<Path<mpfr::mpreal> > paths(n, Path<mpfr::mpreal>(x));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < x; j++) {
			paths[i].set(j, generator() % 15);
		}
	}
	kmeans::iterative_LBG(paths, k);

	return 0;
}