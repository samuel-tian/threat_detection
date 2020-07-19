#include <iostream>
#include <stdio.h>
#include <vector>
#include <gmp.h>
#include <mpfr.h>
#include "mpreal.h"

namespace kmeans {

	/**
	 * Initialize trajectory with number of segments
	 *
	 * @param sz number of segments
	 */
	template<typename T>
	traj<T>::traj(int sz) {
		this->val.resize(sz);
	}	

	/**
	 * Initialize trajectory with number of segments and number of parameters per segment
	 *
	 * @param sz1 number of segments
	 * @param sz2 number of parameters per segment
	 */
	template<typename T>
	traj<T>::traj(int sz1, int sz2) {
		this->val.resize(sz1, kmeans::param<T>(sz2));
	}

	/**
	 * Initialize trajectory with vector of segments and parameters
	 *
	 * @param val vector of segments and parameters
	 */
	template<typename T>
	traj<T>::traj(std::vector<kmeans::param<T> >& val) {
		this->val.resize(val.size());
		for (int i = 0; i < val.size(); i++) {
			this->val.set(i, val[i]);
		}
	}

	/**
	 * Get number of segments in trajectory
	 *
	 * @return number of segments
	 */
	template<typename T>
	int traj<T>::size() {
		return this->val.size();
	}

	/**
	 * Get parameterization at index
	 *
	 * @param i desired index
	 * @return parameterization of ith segment
	 */
	template<typename T>
	param<T> traj<T>::get(int i) {
		return this->val[i];
	}

	/**
	 * Get parameter at index
	 *
	 * @param i desired segment
	 * @param j desired index of parameter in segment
	 * @return parameter at desired index
	 */
	template<typename T>
	T traj<T>::get(int i, int j) {
		return this->val[i].get(j);
	}

	/**
	 * Set parameterization at index
	 *
	 * @param i desired segment
	 * @param v parameterization to be set
	 * @return void
	 */
	template<typename T>
	void traj<T>::set(int i, param<T>& v) {
		this->val[i] = v;
	}

	/**
	 * Set parameter at index
	 *
	 * @param i desired segment
	 * @param j desired index of parameter in segment
	 * @param v parameter to be set
	 * @return void
	 */
	template<typename T>
	void traj<T>::set(int i, int j, T& v) {
		this->val[i].set(j, v);
	}

	/**
	 * Set one trajectory to another
	 *
	 * @param t desired trajectory
	 * @return void
	 */
	template<typename T>
	void traj<T>::operator = (traj<T>& t) {
		this->val.resize(t.size());
		for (int i = 0; i < t.size(); i++) {
			this->val[i] = t.get(i);
		}
	}

	/**
	 * Pass trajectory to outputstream
	 *
	 * @param os outputstream that trajectory will be passed to
	 * @param t trajectory that is being passed to outputstream
	 * @return outputstream
	 */
	template<typename T>
	std::ostream& operator << (std::ostream& os, traj<T>& t) {
		for (int i = 0; i < t.size(); i++) {
			os << t.get(i) << '\n';
		}
		return os;
	}

}