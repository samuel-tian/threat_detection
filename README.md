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
* Boost Template Libraries
```
brew install boost
```

## Compiling Instructions

### No Project Dependencies

Compile:
```
g++ hmm.cpp -o hmm -std=c++17 -lmpfr -lgmp -lgmpxx
```
Execute:
```
./hmm
```

### Generate Static Library / Archive

Compile object files:
```
g++ -c ./hmm/hmm.cpp -std=c++17
g++ -c ./kmeans/kmeans.cpp -std=c++17
```
Compile Archive:
```
ar rsv archive.a ./hmm/hmm.o ./kmeans/kmeans.o
```
Use Archive:
```
g++ -o detect detect.cpp -std=c++17 -lgmp -lmpfr archive.a
```