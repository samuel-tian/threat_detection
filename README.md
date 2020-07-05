# Hidden Markov Models and Computation with Words in Threat Detection Systems

## Dependencies
This project has several dependencies, which are listed below:
* GNU multiple precision arithmetic library (GMP)
```
brew install gmp
```
* GNU multiple precision floating-point reliable library (MPFR)
```
brew install mpfr
```
* MPFR C++: [download link](http://www.holoborodko.com/pavel/mpfr/#download)
	* copy mpreal.h into working directory
## Compiling Instructions
Compile:
```
g++ hmm.cpp -o hmm -std=c++11 -lmpfr -lgmp -lgmpxx
```
Execute:
```
./hmm
```
