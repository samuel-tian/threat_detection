#include <iostream>
#include <stdio.h>
#include <vector>
#include <gmp.h>
#include <mpfr.h>
#include "mpreal.h"
#include <utility>

namespace kmeans {
	
	/**
	 * Initialize parameterization with specified size
	 *
	 * @param sz dimension of parameterization
	 */
	template<typename T>
	param<T>::param(int sz) {
		this->val.resize(sz);
	}

	/**
	 * Initialize parameterization with vector
	 *
	 * @param val vector of parameters
	 */
	template<typename T>
	param<T>::param(std::vector<T>& val) {
		this->val.resize(val.size());
		for (int i = 0; i < val.size(); i++) {
			this->val[i] = val[i];
		}
	}

	/**
	 * Get number of dimensions of parameterization
	 *
	 * @return number of dimensions of paramaterization
	 */
	template<typename T>
	int param<T>::size() {
		return this->val.size();
	}

	/**
	 * Get parameter at index
	 *
	 * @param i desired return index
	 * @return parameter at index i
	 */
	template<typename T>
	T& param<T>::get(int i) {
		return this->val[i];
	}

	/**
	 * Set parameter at index
	 *
	 * @param i desired set index
	 * @param v value to set at index i
	 * @return void
	 */
	template<typename T>
	void param<T>::set(int i, T& v) {
		this->val[i] = v;
	}

	/**
	 * Get centroid of trajectory
	 *
	 * @return pair of (x, y) coordinates representing the location of the centroid
	 */
	template<typename T>
	std::pair<T, T> param<T>::get_centroid() {
		return std::make_pair(this->get(0), this->get(1));
	}

	/**
	 * Set one parameterization to another
	 *
	 * @param p desired paramterization
	 * @return void
	 */
	template<typename T>
	void param<T>::operator = (param<T>& p) {
		this->val.resize(p.size());
		for (int i = 0; i < p.size(); i++) {
			this->val[i] = p.get(i);
		}
	}

	/**
	 * Pass parameterization to an outputstream
	 *
	 * @param os outputstream that parameterization will be passed to
	 * @param p parameterization that is being passed to outputstream
	 * @return outputstream
	 */
	template<typename T>
	std::ostream& operator << (std::ostream& os, param<T>& p) {
		for (int i = 0; i < p.size(); i++) {
			if (i == 0)
				os << "(" << p.get(i);
			else
				os << ", " << p.get(i);
		}
		os << ")";
		return os;
	}

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

	

}
