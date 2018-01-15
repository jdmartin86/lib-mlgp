# lib-mlgp: Library for Most Likely Heterosecastic Gaussian Process Regression

This library implements the Most Likely Heteroscedastic Gaussian Process algorithm, described in

Most Likely Heteroscedastic Gaussian Process Regression, K. Kersting, et. al, 2007

Work began on July 2017
First release was January 2018

### Installing

```
mkdir build
cd build/
cmake ..
make
```

## Running the tests

Pass the cmake flag BUILD_TESTS to build the tests

```
./tests/test-gptd
```
## Built With

* [cmake](https://cmake.org) - To build the library
* [googletest](https://github.com/google/googletest) - Test framework

## Contributing

Please report any bugs or issues to jmarti3@stevens.edu

## Authors

* **John Martin Jr.** - *Creator* - [jdmartin86](https://github.com/jdmartin86)

## Acknowledgments

* Much of the library's code is owed to Manuel Blum's [libgp](https://github.com/mblum/libgp). For the most part, the original structure and approach to Cholesky decompositions is the same as libgp. The covariance code, which was by far the most appealing part of starting from libgp, was borrowed entirely and has been left untouched. The differences are primarily in the prediction routines, which were needed to support heteroscedastic noise. Additionally, I substituted the original optimizers in favor of a header-only L-BFGS library. I made an effort to acknowledge the code Blum wrote by leaving his name and comments in files I didn't touch. For the files I modified significantly, I claim authorship. All questions related to those should be directed to me, since I am most privy to their idiosyncracies.  

* Optimization library: [L-BFGS++](https://github.com/yixuan/LBFGSpp).
