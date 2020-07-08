#include <vector>
#include <gmp.h>
#include <mpfr.h>
#include "mpreal.h"

#ifndef KMEANS_H
#define KMEANS_H

namespace kmeans {

	template<typename T>
	struct Path;

	template<typename T>
	T dif(Path<T>& a, Path<T>& b);

	template<typename T>
	struct Path {
		std::vector<T> parameters;
		Path() = default;
		Path(int s) {
			this->parameters.resize(s);
		}
		Path(std::vector<T> p) {
			this->parameters.resize(p.size());
			for (int i = 0; i < p.size(); i++) {
				this->parameters[i] = p[i];
			}
		}
		size_t size() {
			return this->parameters.size();
		}
		T& operator [] (size_t i) {
			return this->parameters[i];
		}
		void set(int i, T val) {
			this->parameters[i] = val;
		}
		friend T dif(Path& a, Path& b);
		void operator = (Path& p) {
			if (this->size() == p.size()) {
				for (int i = 0; i < this->size(); i++) {
					this->parameters[i] = p[i];
				}
			}
		}
		Path operator += (Path& p) {
			if (p.size() == this->size()) {
				for (int i = 0; i < this->size(); i++) {
					(*this)[i] += p[i];
				}
			}
			return *this;
		}
		Path operator /= (mpfr::mpreal& p) {
			for (int i = 0; i < this->size(); i++) {
				(*this)[i] /= p;
			}
			return *this;
		}
		friend std::ostream& operator << (std::ostream& os, Path& p) {
			for (int i = 0; i < p.size(); i++) {
				if (i == 0)
					os << "(";
				if (i == p.size()-1)
					os << p[i] << ")";
				else
					os << p[i] << ", ";
			}
			return os;
		}
	};


	mpfr::mpreal epsilon;

	void set_threshold(mpfr::mpreal epsilon);

	mpfr::mpreal compute_distortion(std::vector<Path<mpfr::mpreal> >& clusters, std::vector<std::vector<Path<mpfr::mpreal> > >& paths);

	std::vector<Path<mpfr::mpreal> > initialize_clusters(std::vector<Path<mpfr::mpreal> >& paths, int k);

	std::vector<std::vector<Path<mpfr::mpreal> > > recluster(std::vector<Path<mpfr::mpreal> >& clusters, std::vector<Path<mpfr::mpreal> >& paths);

	void update_clusters(std::vector<Path<mpfr::mpreal> >& clusters, std::vector<std::vector<Path<mpfr::mpreal> > >& clustered_paths);

	std::pair<std::vector<Path<mpfr::mpreal> >, std::vector<std::vector<Path<mpfr::mpreal> > > > iterative_LBG(std::vector<Path<mpfr::mpreal> >& paths, int k);

}

#endif // KMEANS_H