#include <vector>
#include <gmp.h>
#include <mpfr.h>
#include "mpreal.h"

#ifndef KMEANS_H
#define KMEANS_H

namespace kmeans {

	template<typename T>
	struct Path {
		std::vector<T> parameters;
		Path() = default;
		Path(std::vector<T> p) {
			this->parameters.resize(p.size());
		}
		size_t size() {
			return paramters.size();
		}
		T& operator [] (size_t i) {
			return this->paramters[i];
		}
		Path operator += (Path const& p) {
			if (p.size() == this->size()) {
				for (int i = 0; i < this->size(); i++) {
					(*this)[i] += p[i];
				}
			}
			return *this;
		}
		Path operator /= (mpfr::mpreal const& p) {
			if (p.size() == this->size()) {
				for (int i = 0; i < this->size(); i++) {
					(*this)[i] /= p;
				}
			}
			return *this;
		}
		friend std::ostream& operator << (std::ostream os, Path const& p) {
			for (int i = 0; i < p.size(); i++) {
				os << p[i] << " ";
			}
			os << '\n';
			return os;
		}
	};

	mpfr::mpreal epsilon;

	mpfr::mpreal compute_distortion(std::vector<Path<mpfr::mpreal> >& clusters, std::vector<Path<mpfr::mpreal> >& paths);

	std::vector<Path<mpfr::mpreal> > initialize_clusters(std::vector<Path<mpfr::mpreal> >& paths);

	void recluster(std::vector<Path<mpfr::mpreal> >& clusters, std::vector<Path<mpfr::mpreal> >& paths);

	void iterative_LBG(std::vector<Path<mpfr::mpreal> >& paths);

}

#endif // KMEANS_H