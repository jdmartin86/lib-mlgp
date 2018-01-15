INCLUDE(ExternalProject)

# ----- Download and build gtest -----
ExternalProject_Add(googletest
  GIT_REPOSITORY "https://github.com/google/googletest.git"
  GIT_TAG "release-1.7.0"
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
  PREFIX "gtest"
)

FIND_PACKAGE(Threads REQUIRED)

# ----- Testing -----
INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/gtest/src/googletest/include)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/gtest/src/googletest-build)

# ----- Add sources -----
SET(LIBMLGP_TESTS 
  test-gp_regression.cpp
  test-hetero_gp_regression.cpp
  test-log_likelihood.cpp
  test-covariance_functions.cpp
  test-gp_utils.cpp
  test-cov_factory.cpp
)

ADD_EXECUTABLE(gptest ${LIBMLGP_TESTS})
TARGET_LINK_LIBRARIES(gptest mlgp ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main)
