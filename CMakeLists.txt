#
# CMake file for lib-mlgp
#
# John Martin Jr.
# jmarti3@stevens.edu
# Copyright 2017
#
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT(lib-mlgp)

# colors for messages
IF(NOT WIN32)
  STRING(ASCII 27 Esc)
  SET(ColorReset "${Esc}[m")
  SET(ColorBold  "${Esc}[1m")
  SET(Red         "${Esc}[31m")
  SET(Green       "${Esc}[32m")
  SET(Yellow      "${Esc}[33m")
  SET(Blue        "${Esc}[34m")
  SET(Magenta     "${Esc}[35m")
  SET(Cyan        "${Esc}[36m")
  SET(White       "${Esc}[37m")
  SET(BoldRed     "${Esc}[1;31m")
  SET(BoldGreen   "${Esc}[1;32m")
  SET(BoldYellow  "${Esc}[1;33m")
  SET(BoldBlue    "${Esc}[1;34m")
  SET(BoldMagenta "${Esc}[1;35m")
  SET(BoldCyan    "${Esc}[1;36m")
  SET(BoldWhite   "${Esc}[1;37m")
ENDIF()
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fPIC -pthread")

OPTION(BUILD_TESTS "Build tests" ON)

FIND_PACKAGE(Eigen3 3.0.1 REQUIRED)
IF( NOT( Eigen_FOUND ) )
  MESSAGE( "${Red}-- Failed to find the Eigen library${ColorReset}" )
ENDIF( )

# ----- Add include directories -----
INCLUDE_DIRECTORIES(SYSTEM ${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----- Add sources -----
SET(LIBMLGP_SRC
  src/cov.cc
  src/cov_factory.cc
  src/cov_linear_ard.cc
  src/cov_linear_one.cc
  src/cov_matern3_iso.cc
  src/cov_matern5_iso.cc
  src/cov_noise.cc
  src/cov_rq_iso.cc
  src/cov_periodic_matern3_iso.cc
  src/cov_periodic.cc
  src/cov_se_ard.cc
  src/cov_se_iso.cc
  src/cov_sum.cc
  src/cov_prod.cc
  src/gp.cc
  src/gp_opt.cc
  src/gp_utils.cc
  src/sampleset.cc
  src/input_dim_filter.cc
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/lbfgspp)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/lbfgspp/LBFGS)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----- Add library target -----
ADD_LIBRARY(mlgp ${LIBMLGP_SRC})

# ----- Testing -----
IF(BUILD_TESTS)
  ADD_SUBDIRECTORY(tests)
ENDIF()

# ----- Install -----
INSTALL(TARGETS mlgp DESTINATION lib/lib-mlgp)
INSTALL(DIRECTORY include opt DESTINATION include/lib-mlgp)
