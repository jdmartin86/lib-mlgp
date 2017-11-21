# lib-mlgp: Most Likely Heterosecastic Gaussian Process Library

This library implements the Most Likely Heteroscedastic Gaussian Process algorithm, described in

Most Likely Heteroscedastic Gaussian Process Regression, K. Kersting, et. al, 2007

July. 2017

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
cmake .. -DBUILD_TESTS:BOOL=TRUE
make
./tests/test-gptd
```
## Built With

* [cmake](https://cmake.org) - To build the library
* [googletest](https://github.com/google/googletest) - Test framework

## Contributing

No external contributions are allowed at this time. This will change after the first software release.

## Authors

* **John Martin Jr.** - *Creator* - [jdmartin86](https://github.com/jdmartin86)

## Acknowledgments

* Forked originally from Manuel Blum's [libgp](https://github.com/mblum/libgp).
