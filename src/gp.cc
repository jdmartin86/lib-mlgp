// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "cov_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {
  
  const double log2pi = log(2*M_PI);
  const double initial_L_size = 1000;

  GaussianProcess::GaussianProcess ()
  {
      sampleset_f = NULL;
      sampleset_g = NULL;
      cf = NULL;      
      cg = NULL;
  }

  GaussianProcess::GaussianProcess (size_t input_dim, std::string covf_def)
  {
    // set input dimensionality
    this->input_dim = input_dim;
    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    cf->loghyper_changed = 0;
    sampleset_f = new SampleSet(input_dim);
    L_f.resize(initial_L_size, initial_L_size);
  }

  GaussianProcess::GaussianProcess( size_t input_dim, 
				    std::string covf_def,
				    std::string covg_def )
  {
    // set input dimensionality
    this->input_dim = input_dim;
    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    cg = factory.create(input_dim, covg_def);
    cf->loghyper_changed = 0;
    cg->loghyper_changed = 0;
    sampleset_f = new SampleSet(input_dim);
    sampleset_g = new SampleSet(input_dim);
    L_f.resize(initial_L_size, initial_L_size);
    L_g.resize(initial_L_size, initial_L_size);
  }
  
  GaussianProcess::GaussianProcess (const char * filename) 
  {
    sampleset_f = NULL;
    cf = NULL;
    int stage = 0;
    std::ifstream infile;
    double y;
    infile.open(filename);
    std::string s;
    double * x = NULL;
    L_f.resize(initial_L_size, initial_L_size);
    while (infile.good()) {
      getline(infile, s);
      // ignore empty lines and comments
      if (s.length() != 0 && s.at(0) != '#') {
        std::stringstream ss(s);
        if (stage > 2) {
          ss >> y;
          for(size_t j = 0; j < input_dim; ++j) {
            ss >> x[j];
          }
          add_pattern(x, y);
        } else if (stage == 0) {
          ss >> input_dim;
          sampleset_f = new SampleSet(input_dim);
          x = new double[input_dim];
        } else if (stage == 1) {
          CovFactory factory;
          cf = factory.create(input_dim, s);
          cf->loghyper_changed = 0;
        } else if (stage == 2) {
          Eigen::VectorXd params(cf->get_param_dim());
          for (size_t j = 0; j<cf->get_param_dim(); ++j) {
            ss >> params[j];
          }
          cf->set_loghyper(params);
        }
        stage++;
      }
    }
    infile.close();
    if (stage < 3) {
      std::cerr << "fatal error while reading " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    delete [] x;
  }
  
  GaussianProcess::GaussianProcess(const GaussianProcess& gp)
  {
    this->input_dim = gp.input_dim;
    sampleset_f = new SampleSet(*(gp.sampleset_f));
    alpha_f = gp.alpha_f;
    k_star_f = gp.k_star_f;
    alpha_needs_update = gp.alpha_needs_update;
    L_f = gp.L_f;
    
    // copy covariance function
    CovFactory factory;
    cf = factory.create(gp.input_dim, gp.cf->to_string());
    cf->loghyper_changed = gp.cf->loghyper_changed;
    cf->set_loghyper(gp.cf->get_loghyper());
  }
  
  GaussianProcess::~GaussianProcess ()
  {
    // free memory
    if (sampleset_f != NULL) delete sampleset_f;
    if (cf != NULL) delete cf;
  }  
  
  double GaussianProcess::f(const double x[])
  {
    if (sampleset_f->empty()) return 0;
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    compute();
    update_alpha();
    update_k_star(x_star);
    return k_star_f.dot(alpha_f);    
  }
  
  double GaussianProcess::var(const double x[])
  {
    if (sampleset_f->empty()) return 0;
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    compute();
    update_alpha();
    update_k_star(x_star);
    int n = sampleset_f->size();
    Eigen::VectorXd v = L_f.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star_f);
    return cf->get(x_star, x_star) - v.dot(v);	
  }

  void GaussianProcess::compute()
  {
    // can previously computed values be used?
    if (!cf->loghyper_changed) return;
    cf->loghyper_changed = false;
    int n = sampleset_f->size();
    // resize L if necessary
    if (n > L_f.rows()) L_f.resize(n + initial_L_size, n + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset_f->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        L_f(i, j) = cf->get(sampleset_f->x(i), sampleset_f->x(j));
      }
    }
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    L_f.topLeftCorner(n, n) = L_f.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
    alpha_needs_update = true;
  }
  
  void GaussianProcess::update_k_star(const Eigen::VectorXd &x_star)
  {
    k_star_f.resize(sampleset_f->size());
    for(size_t i = 0; i < sampleset_f->size(); ++i) {
      k_star_f(i) = cf->get(x_star, sampleset_f->x(i));
    }
  }

  void GaussianProcess::update_alpha()
  {
    // can previously computed values be used?
    if (!alpha_needs_update) return;
    alpha_needs_update = false;
    alpha_f.resize(sampleset_f->size());
    // Map target values to VectorXd
    const std::vector<double>& targets = sampleset_f->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset_f->size());
    int n = sampleset_f->size();
    alpha_f = L_f.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y);
    L_f.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha_f);
  }
  
  void GaussianProcess::add_pattern(const double x[], double y)
  {
    //std::cout<< L_f.rows() << std::endl;
#if 0
    sampleset_f->add(x, y);
    cf->loghyper_changed = true;
    alpha_needs_update = true;
    cached_x_star = NULL;
    return;
#else
    int n = sampleset_f->size();
    sampleset_f->add(x, y);
    // create kernel matrix if sampleset_f is empty
    if (n == 0) {
      L_f(0,0) = sqrt(cf->get(sampleset_f->x(0), sampleset_f->x(0)));
      cf->loghyper_changed = false;
    // recompute kernel matrix if necessary
    } else if (cf->loghyper_changed) {
      compute();
    // update kernel matrix 
    } else {
      Eigen::VectorXd k(n);
      for (int i = 0; i<n; ++i) {
        k(i) = cf->get(sampleset_f->x(i), sampleset_f->x(n));
      }
      double kappa = cf->get(sampleset_f->x(n), sampleset_f->x(n));
      // resize L_f if necessary
      if (sampleset_f->size() > static_cast<std::size_t>(L_f.rows())) {
        L_f.conservativeResize(n + initial_L_size, n + initial_L_size);
      }
      L_f.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
      L_f.block(n,0,1,n) = k.transpose();
      L_f(n,n) = sqrt(kappa - k.dot(k));
    }
    alpha_needs_update = true;
#endif
  }

  bool GaussianProcess::set_y(size_t i, double y) 
  {
    if(sampleset_f->set_y(i,y)) {
      alpha_needs_update = true;
      return 1;
    }
    return false;
  }

  size_t GaussianProcess::get_sampleset_size()
  {
    return sampleset_f->size();
  }
  
  void GaussianProcess::clear_sampleset()
  {
    sampleset_f->clear();
  }
  
  void GaussianProcess::write(const char * filename)
  {
    // output
    std::ofstream outfile;
    outfile.open(filename);
    time_t curtime = time(0);
    tm now=*localtime(&curtime);
    char dest[BUFSIZ]= {0};
    strftime(dest, sizeof(dest)-1, "%c", &now);
    outfile << "# " << dest << std::endl << std::endl
    << "# input dimensionality" << std::endl << input_dim << std::endl 
    << std::endl << "# covariance function" << std::endl 
    << cf->to_string() << std::endl << std::endl
    << "# log-hyperparameter" << std::endl;
    Eigen::VectorXd param = cf->get_loghyper();
    for (size_t i = 0; i< cf->get_param_dim(); i++) {
      outfile << std::setprecision(10) << param(i) << " ";
    }
    outfile << std::endl << std::endl 
    << "# data (target value in first column)" << std::endl;
    for (size_t i=0; i<sampleset_f->size(); ++i) {
      outfile << std::setprecision(10) << sampleset_f->y(i) << " ";
      for(size_t j = 0; j < input_dim; ++j) {
        outfile << std::setprecision(10) << sampleset_f->x(i)(j) << " ";
      }
      outfile << std::endl;
    }
    outfile.close();
  }
  
  CovarianceFunction & GaussianProcess::covf()
  {
    return *cf;
  }
  
  size_t GaussianProcess::get_input_dim()
  {
    return input_dim;
  }

  double GaussianProcess::log_likelihood()
  {
    compute();
    update_alpha();
    int n = sampleset_f->size();
    const std::vector<double>& targets = sampleset_f->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset_f->size());
    double det = 2 * L_f.diagonal().head(n).array().log().sum();
    return -0.5*y.dot(alpha_f) - 0.5*det - 0.5*n*log2pi;
  }

  Eigen::VectorXd GaussianProcess::log_likelihood_gradient() 
  {
    compute();
    update_alpha();
    size_t n = sampleset_f->size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

    // compute kernel matrix inverse
    L_f.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
    L_f.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(W);

    W = alpha_f * alpha_f.transpose() - W;

    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j <= i; ++j) {
        cf->grad(sampleset_f->x(i), sampleset_f->x(j), g);
        if (i==j) grad += W(i,j) * g * 0.5;
        else      grad += W(i,j) * g;
      }
    }

    return grad;
  }
}
