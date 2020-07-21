#include <vector>
#include <gmp.h>
#include <mpfr.h>
#include "mpreal.h"
#include <utility>
#include <boost/multiprecision/mpfr.hpp>

#ifndef KMEANS_H
#define KMEANS_H

using namespace boost::multiprecision;

namespace kmeans {

	/**
	 * parameter class
	 *
	 * vector of numeric values that represents the Fourier parameterization of a trajectory
	 */
	template<typename T>
	class param {
	public:
		std::vector<T> val;
		param() = default;
		param(int sz);
		param(std::vector<T>& val);
		int size();
		T& get(int i);
		void set(int i, T& v);
		std::pair<T, T> get_centroid();
		void operator = (param p);
		template<typename S>
		friend std::ostream& operator << (std::ostream& os, param<S> p);
		template<typename S>
		friend S difference(param<S>& a, param<S>& b);
	};

	/**
	 * trajectory class
	 *
	 * vector of parameterizations that represents the segments of the trajectories
	 */
	template<typename T>
	class traj {
	public:
		std::vector<kmeans::param<T> > val;
		traj() = default;
		traj(int sz);
		traj(int sz1, int sz2);
		traj(std::vector<kmeans::param<T> >& val);
		int size();
		param<T> get(int i);
		T get(int i, int j);
		void set(int i, param<T>& v);
		void set(int i, int j, T& v);
		void operator = (traj t);
		template<typename S>
		friend std::ostream& operator << (std::ostream& os, traj<S> t);
	};

	extern mpfr_float epsilon;
	void set_threshold(mpfr_float e);
	mpfr_float get_threshold();
	mpfr_float compute_distortion(std::vector<param<mpfr_float> >& centroids, std::vector<std::vector<param<mpfr_float> > >& clustered_paths);
	std::vector<param<mpfr_float> > initialize_centroids(std::vector<param<mpfr_float> >& paths, int k);
	std::vector<std::vector<param<mpfr_float> > > recluster(std::vector<param<mpfr_float> >& centroids, std::vector<param<mpfr_float> >& );
	void update_clusters(std::vector<param<mpfr_float> >& centroids, std::vector<std::vector<param<mpfr_float> > >& clustered_paths);
	int nearest_centroid(std::vector<param<mpfr_float> >& centroids, param<mpfr_float>& path);
	std::vector<param<mpfr_float> > iterative_LBG(std::vector<param<mpfr_float> >& paths, int k);

}

#include "param.tpp"
#include "traj.tpp"

#endif // KMEANS_H