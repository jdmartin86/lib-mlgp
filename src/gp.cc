// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "cov_factory.h"
#include "cg.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

#define SQ( x ) ( ( x ) * ( x ) )

namespace libgp {
  
  const double log2pi = log(2*M_PI);
  const double initial_L_size = 1000;
  const double num_var_samp = 100.0;

  /**
   * GaussianProcess
   * 
   * Create an empty instance of a heteroscedastic Gaussian process.
   */
  GaussianProcess::GaussianProcess( )
  {
    sampleset = NULL;
    cf = NULL;
    h = NULL;
  }

  /**
   * GaussianProcess
   * 
   * Create an instance of a homoscedastic Gaussian process, grounded
   * with a covariance definition.
   */
  GaussianProcess::GaussianProcess( size_t input_dim, std::string covf_def )
  {
    // set input dimensionality
    this->input_dim = input_dim;

    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    cf->loghyper_changed = 0;
    sampleset = new SampleSet(input_dim);
    L.resize(initial_L_size, initial_L_size);
    this->h = NULL;
  }

  /**
   * GaussianProcess
   * 
   * Create an instance of a heteroscedastic Gaussian process, grounded
   * with a covariance definitions.
   */
  GaussianProcess::GaussianProcess( size_t input_dim, 
				    std::string covf_def,
				    std::string covh_def )
  {
    // set input dimensionality
    this->input_dim = input_dim;

    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    cf->loghyper_changed = 0;
    sampleset = new SampleSet(input_dim);
    L.resize(initial_L_size, initial_L_size);

    // create the logarithmic noise process
    h = new GaussianProcess( input_dim , covh_def );
  }
  
  /**
   * GaussianProcess
   * 
   * Create an instance of a homoscedastic Gaussian process from 
   * a file.
   */
  GaussianProcess::GaussianProcess( const char* filename ) 
  {
    h = NULL;
    sampleset = NULL;
    cf = NULL;
    int stage = 0;
    std::ifstream infile;
    double y;
    infile.open(filename);
    std::string s;
    double * x = NULL;
    L.resize(initial_L_size, initial_L_size);
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
          sampleset = new SampleSet(input_dim);
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

  /**
   * GaussianProcess
   * 
   * Copy an instance of a homoscedastic Gaussian process
   */
  GaussianProcess::GaussianProcess( const GaussianProcess& gp )
  {
    this->input_dim = gp.input_dim;
    sampleset = new SampleSet(*(gp.sampleset));
    alpha = gp.alpha;
    k_star = gp.k_star;
    alpha_needs_update = gp.alpha_needs_update;
    L = gp.L;
    h = NULL;

    // copy covariance function
    CovFactory factory;
    cf = factory.create(gp.input_dim, gp.cf->to_string());
    cf->loghyper_changed = gp.cf->loghyper_changed;
    cf->set_loghyper(gp.cf->get_loghyper());
  }
  
  /**
   * ~GaussianProcess
   * 
   * Discard a Gaussian process class and free any allocated memory
   */
  GaussianProcess::~GaussianProcess( )
  {
    // free memory
    if(sampleset != NULL) delete sampleset;
    if(cf != NULL) delete cf;
    if( h != NULL ) delete h;
  }  
  
  /**
   * f
   * 
   * Compute the predictive mean value at test input x_star
   */
  double GaussianProcess::f( const Eigen::VectorXd& x_star )
  {
    if( sampleset->empty() ) return 0.0;
    if( h != NULL ) update_noise( );
    compute();
    update_alpha();
    update_k_star(x_star);
    return k_star.dot(alpha);    
  }

  double GaussianProcess::f( const double x[] )
  {
    Eigen::Map<const Eigen::VectorXd> x_star( x , input_dim );
    return f( x_star );    
  }

  double GaussianProcess::var( const Eigen::VectorXd& x_star )
  {
    if (sampleset->empty()) return 0;
    if( h != NULL ) update_noise( );
    compute();
    update_alpha();
    update_k_star(x_star);
    int n = sampleset->size();
    Eigen::VectorXd v = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star);
    return cf->get(x_star, x_star) - v.dot(v);	
  }

  double GaussianProcess::var(const double x[])
  {
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    return var( x_star );	
  }


  /**
   * compute
   *
   * Compute the Cholesky decomp
   */
  void GaussianProcess::compute()
  {
    // can previously computed values be used?
    if (!cf->loghyper_changed) return;
    cf->loghyper_changed = false;
    int n = sampleset->size();
    // resize L if necessary
    if (n > L.rows()) L.resize(n + initial_L_size, n + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        L(i,j) = cf->get(sampleset->x(i), sampleset->x(j));
      }
      if( h != NULL ) L(i,i) += h->f( sampleset->x(i) );
    }
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    L.topLeftCorner(n, n) = L.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
    alpha_needs_update = true;
  }
  
  void GaussianProcess::update_k_star(const Eigen::VectorXd &x_star)
  {
    k_star.resize(sampleset->size());
    for(size_t i = 0; i < sampleset->size(); ++i) {
      k_star(i) = cf->get(x_star, sampleset->x(i));
    }
  }

  void GaussianProcess::update_alpha()
  {
    // can previously computed values be used?
    if (!alpha_needs_update) return;
    alpha_needs_update = false;
    alpha.resize(sampleset->size());
    // Map target values to VectorXd
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    int n = sampleset->size();
    alpha = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha);
  }

  /**
   * update_noise
   * 
   * Update the h-process data set with maximum likelihood predictions
   */
  void GaussianProcess::update_noise( )
  {
    if( !noise_needs_update ) return; 
    std::random_device rd;
    std::mt19937 gen(rd());
    double f_i = 0.0 , v_i = 0.0 , z_i = 0.0;

    // TODO: test convergence
    for( size_t k = 0 ; k < 10 ; k++ )
    {
      // update h-process dataset
      for( size_t i = 0 ; i < sampleset->size() ; ++i ) 
      {
	f_i = f( sampleset->x(i) );
	v_i = var( sampleset->x(i) );
      
	std::normal_distribution<double> N( f_i , v_i );
	z_i = 0.0;
	for( size_t j = 0 ; j < (size_t) num_var_samp ; j++ ) 
	  z_i += 0.5 * SQ( sampleset->y(i) - N(gen) );
	z_i = log( z_i / num_var_samp );
	h->sampleset->set_y( i , z_i ); 
      }
    
      // optimize f-process hyperparameters
      CG cg;
      cg.maximize( this , 50 , 0 );
    }
    noise_needs_update = false;
  }
  
  /**
   * add_pattern
   *
   * Adds a sample to the training set and updates the h-process set
   * when applicable.
   */
  void GaussianProcess::add_pattern(const double x[], double y)
  {
    sampleset->add( x , y );

    // add sample to h process
    if( h != NULL )
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      double f_i = f( x );
      double v_i = var( x );
      std::normal_distribution<double> N( f_i , v_i );
      double z_i = 0.0;
      for( size_t j = 0 ; j < (size_t) num_var_samp ; j++ ) 
	z_i += 0.5 * SQ( y - N(gen) );
      z_i = log( z_i / num_var_samp );
      h->sampleset->add( x , z_i );
      noise_needs_update = true;
    }

    cf->loghyper_changed = true;
    alpha_needs_update = true;
  }

  bool GaussianProcess::set_y(size_t i, double y) 
  {
    if(sampleset->set_y(i,y)) {
      alpha_needs_update = true;
      return 1;
    }
    return false;
  }

  size_t GaussianProcess::get_sampleset_size()
  {
    return sampleset->size();
  }
  
  void GaussianProcess::clear_sampleset()
  {
    sampleset->clear();
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
    for (size_t i=0; i<sampleset->size(); ++i) {
      outfile << std::setprecision(10) << sampleset->y(i) << " ";
      for(size_t j = 0; j < input_dim; ++j) {
        outfile << std::setprecision(10) << sampleset->x(i)(j) << " ";
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
    int n = sampleset->size();
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    double det = 2 * L.diagonal().head(n).array().log().sum();
    return -0.5*y.dot(alpha) - 0.5*det - 0.5*n*log2pi;
  }

  Eigen::VectorXd GaussianProcess::log_likelihood_gradient() 
  {
    compute();
    update_alpha();
    size_t n = sampleset->size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

    // compute kernel matrix inverse
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(W);

    W = alpha * alpha.transpose() - W;

    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j <= i; ++j) {
        cf->grad(sampleset->x(i), sampleset->x(j), g);
        if (i==j) grad += W(i,j) * g * 0.5;
        else      grad += W(i,j) * g;
      }
    }

    return grad;
  }
}
