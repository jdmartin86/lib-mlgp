// lib-mlgp - Library for Most-likely Hetroscedastic Gaussian Process Regression
// Copyright (c) 2018, John Martin <jmarti3@stevens.edu>
// All rights reserved.

#include "gp.h"
#include "gp_opt.h"
#include "cov_factory.h"
#include "LBFGS.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {
  
  const double log2pi = log(2*M_PI);
  const double initial_L_size = 1000;
  const double num_var_samp = 100.0;

  /**
   * GaussianProcess
   * 
   * Create an empty instance of a Gaussian process.
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
    Lf.resize(initial_L_size, initial_L_size);

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
   * Compute the f-process's predicitve mean
   */
  double GaussianProcess::f( const Eigen::VectorXd& x_star )
  {
    if( sampleset->empty( ) ) return 0.0;
    computef( );
    update_alphaf( );
    update_k_star( x_star );
    return k_star.dot( alphaf );    
  }

  double GaussianProcess::f( const double x[] )
  {
    Eigen::Map<const Eigen::VectorXd> x_star( x , input_dim );
    return f( x_star );    
  }

  /**
   * g
   * 
   * Compute the exponentiated h-process predicitve mean
   */
  double GaussianProcess::g( const Eigen::VectorXd& x_star )
  {
    if( h == NULL ) return 0.0;
    return exp( h->f( x_star ) );    
  }

  double GaussianProcess::g( const double x[] )
  {
    Eigen::Map<const Eigen::VectorXd> x_star( x , input_dim );
    return g( x_star );    
  }

  /**
   * mean
   * 
   * Compute the composit process's predicitve mean
   */
  double GaussianProcess::mean( const Eigen::VectorXd& x_star )
  {
    if( sampleset->empty( ) ) return 0.0;
    if( h != NULL ) update_noise( );
    compute( );
    update_alpha( );
    update_k_star( x_star );
    return k_star.dot( alpha );    
  }

  double GaussianProcess::mean( const double x[] )
  {
    Eigen::Map<const Eigen::VectorXd> x_star( x , input_dim );
    return mean( x_star );    
  }

  /**
   * varf
   *
   * Compute the variance of the predictive posterior
   */
  double GaussianProcess::varf( const Eigen::VectorXd& x_star )
  {
    if( sampleset->empty( ) ) return 0;
    computef();
    update_alphaf();
    update_k_star(x_star);
    int n = sampleset->size();
    Eigen::VectorXd v = Lf.topLeftCorner(n, n).
      triangularView<Eigen::Lower>().solve(k_star);
    return cf->get(x_star, x_star) - v.dot(v);	
  }

  double GaussianProcess::varf( const double x[] )
  {
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    return varf( x_star );	
  }

  /**
   * varg
   *
   * Compute the variance of the exponentiated h-process predictive posterior
   */
  double GaussianProcess::varg( const Eigen::VectorXd& x_star )
  {
    if( h == NULL ) return 0.0;
    return exp( h->var( x_star ) ) ;    
  }

  double GaussianProcess::varg( const double x[] )
  {
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    return varg( x_star );	
  }

  /**
   * var
   *
   * Compute the variance of the predictive posterior
   */
  double GaussianProcess::var( const Eigen::VectorXd& x_star )
  {
    if( sampleset->empty( ) ) return 0;
    if( h != NULL ) update_noise( );
    compute();
    update_alpha();
    update_k_star(x_star);
    int n = sampleset->size();
    Eigen::VectorXd v = L.topLeftCorner(n, n).
      triangularView<Eigen::Lower>().solve(k_star);
    return cf->get(x_star, x_star) - v.dot(v);	
  }

  double GaussianProcess::var( const double x[] )
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
    if( h != NULL )
    { 
      if(!cf->loghyper_changed && !h->cf->loghyper_changed) return;
    }
    else
    {
      if(!cf->loghyper_changed) return;
    }

    cf->loghyper_changed = false;
    int n = sampleset->size();
    // resize L if necessary
    if (n > L.rows()) L.resize(n + initial_L_size, n + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        L(i,j) = cf->get(sampleset->x(i), sampleset->x(j));
      }
      if( h != NULL )
      {
	L(i,i) += g( sampleset->x(i) );
      }
    }
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    L.topLeftCorner(n, n) = L.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
    alpha_needs_update = true;
  }

  /**
   * computef
   *
   * Compute the Cholesky decomp of the f-process
   */
  void GaussianProcess::computef()
  {
    // can previously computed values be used?
    if (!cf->loghyper_changed) return;
    cf->loghyper_changed = false;
    int n = sampleset->size();
    // resize L if necessary
    if (n > Lf.rows()) Lf.resize(n + initial_L_size, n + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        Lf(i,j) = cf->get(sampleset->x(i), sampleset->x(j));
      }
    }
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    Lf.topLeftCorner(n, n) = Lf.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
    alphaf_needs_update = true;
  }
  
  /**
   * update_k_star
   *
   * Compute the covariance test vector for the given process
   */
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

  void GaussianProcess::update_alphaf()
  {
    // can previously computed values be used?
    if (!alphaf_needs_update) return;
    alphaf_needs_update = false;
    alphaf.resize(sampleset->size());
    // Map target values to VectorXd
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    int n = sampleset->size();
    alphaf = Lf.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y);
    Lf.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().
      solveInPlace(alphaf);
  }

  /**
   * update_noise
   * 
   * Update the h-process dataset with maximum likelihood predictions
   */
  void GaussianProcess::update_noise( )
  {
    if( !noise_needs_update ) return; 
    noise_needs_update = false;

    std::random_device rd;
    std::mt19937 gen(rd());
    double f_i = 0.0 , v_i = 0.0 , z_i = 0.0;

    Eigen::VectorXd p_prev = covf().get_loghyper();
    Eigen::VectorXd p_curr = covf().get_loghyper();
    Eigen::VectorXd z(sampleset->size());

    for( size_t k = 0 ; k < 50 ; k++ )
    {
      // optimize f-process hyperparameters
      std::cout << "Optimizing f-process: ";    
      optimize_hyper( );

      // update h-process dataset by computing the sample variance
      for( size_t i = 0 ; i < sampleset->size() ; ++i ) 
      {

	// we sample the predictive, but limit the width of the dist
	// at training points; it goes to zero in principle.
	f_i = Utils::bound( mean( sampleset->x(i) ) , -1.0E6 , 1.0E6 );
	v_i = Utils::bound( var( sampleset->x(i) ) , 1.0E-6 , 1.0E6 );
	std::normal_distribution<double> N( f_i , sqrt(v_i) );
	//std::cout << "Sampling from N("<<f_i<<","<<v_i<<")\n";

	z_i = 0.0;
	for( size_t j = 0 ; j < (size_t) num_var_samp ; j++ ) 
	  z_i += pow( sampleset->y(i) - N(gen) , 2.0 );

	//std::cout << "z_i = " << z_i << std::endl;
	z_i = Utils::bound( z_i  , 0.0 , 1.0E6 ) / ( num_var_samp - 1.0 );
	z(i) = log( z_i );
	h->set_y( i , z(i) ); 

      }

      // optimize h-process hyperparameters 
      std::cout << "Optimizing h-process\n";
      optimize_hhyper();

      // optimize composite process hyperparameters 
      std::cout << "Optimizing composite process\n";
      optimize_hyper();

      // check convergence
      p_curr = covf().get_loghyper();
      Eigen::VectorXd diff = p_curr - p_prev;
      if( diff.norm() < 1.0E-2 )
      { 
	std::cout << "Convergence reached!\n";
	break;
      }
      p_prev = p_curr;
    }
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
    cf->loghyper_changed = true;
    alpha_needs_update = true;

    // add sample to h process
    if( h != NULL )
    {
      h->sampleset->add( x , 1.0E-6 );
      noise_needs_update = true;
    }
  }

  bool GaussianProcess::set_y(size_t i, double y) 
  {
    if(sampleset->set_y(i,y)) {
      alpha_needs_update = true;
      return 1;
    }
    return false;
  }

  double GaussianProcess::get_z( size_t i )
  {
    if( h != NULL ) return h->sampleset->y(i);
    return 0.0;
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

  CovarianceFunction & GaussianProcess::covh()
  {
    return h->covf();
  }

  /**
   * draw_random_hetero_sample
   *
   * Draws a random sample from the noise process prior, then uses that to draw 
   * a sample from the f-process prior.
   */
  Eigen::VectorXd GaussianProcess::draw_random_hetero_sample( Eigen::MatrixXd& X,
							      Eigen::VectorXd& z)
  {
    assert (X.cols() == int(input_dim)); 
    int n = X.rows();

    if( h == NULL ) return z;

    // draw a random function from the h-process
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> N( 0.0 , 0.1 );

    Eigen::MatrixXd K(n, n);
    Eigen::LLT<Eigen::MatrixXd> solver;
    Eigen::VectorXd y(n);
    // compute kernel matrix (lower triangle)
    for(int i = 0; i < n; ++i) {
      for(int j = i; j < n; ++j) {
        K(j, i) = h->covf().get(X.row(j), X.row(i));
      }
      y(i) = N(gen);
    }
    // perform cholesky factorization
    solver = K.llt();  
    z = solver.matrixL() * y;

    // draw random sample from the f-process
    K.setZero();
    // compute kernel matrix (lower triangle)
    for(int i = 0; i < n; ++i) {
      for(int j = i; j < n; ++j) {
        K(j, i) = cf->get(X.row(j), X.row(i));
      }
      K(i,i) += exp( z(i) );
      y(i) = Utils::randn();
    }
    // perform cholesky factorization
    solver = K.llt();  
    return solver.matrixL() * y;

  }

  size_t GaussianProcess::get_input_dim()
  {
    return input_dim;
  }

  double GaussianProcess::log_flikelihood()
  {
    computef();
    update_alphaf();
    int n = sampleset->size();
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    double det = 2 * Lf.diagonal().head(n).array().log().sum();
    return -0.5*y.dot(alphaf) - 0.5*det - 0.5*n*log2pi;
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

  Eigen::VectorXd GaussianProcess::log_flikelihood_gradient() 
  {
    computef();
    update_alphaf();
    size_t n = sampleset->size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

    // compute kernel matrix inverse
    Lf.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
    Lf.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(W);
    W = alphaf * alphaf.transpose() - W;

    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j <= i; ++j) {
        cf->grad(sampleset->x(i), sampleset->x(j), g);
        if (i==j) grad += W(i,j) * g * 0.5;
        else      grad += W(i,j) * g;
      }
    }

    return grad;
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

  double GaussianProcess::norm_log_flikelihood()
  {
    return log_flikelihood() / log_likelihood_baseline;
  }

  double GaussianProcess::norm_log_likelihood()
  {
    return log_likelihood() / log_likelihood_baseline;
  }

  void GaussianProcess::optimize_fhyper( )
  {
    LBFGSpp::LBFGSParam<double> param;
    param.epsilon = 1.0e-3;
    param.max_iterations = 25;
    LBFGSpp::LBFGSSolver<double> solver(param);
    GaussianProcessfOpt fun;    
    fun.parent( this );

    // compute the baseline with which other evaluations are normalized
    log_likelihood_baseline = log_flikelihood();

    Eigen::VectorXd x(covf().get_param_dim());
    x.setZero();
    double fx;
    int niter = solver.minimize( fun , x , fx );
    cf->loghyper_changed = true;

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
  }

  void GaussianProcess::optimize_hhyper( )
  {
    LBFGSpp::LBFGSParam<double> param;
    param.epsilon = 1.0E-3;
    param.past = 2;
    param.delta = 1.0E-3;
    param.max_iterations = 25;
    LBFGSpp::LBFGSSolver<double> solver(param);
    GaussianProcessfOpt fun;    
    fun.parent( h );

    // compute the baseline with which other evaluations are normalized
    h->log_likelihood_baseline = h->log_flikelihood();

    Eigen::VectorXd x(h->covf().get_param_dim());
    x.setZero();
    double fx;
    int niter = solver.minimize( fun , x , fx );
    h->cf->loghyper_changed = true;

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
  }

  void GaussianProcess::optimize_hyper( )
  {
    LBFGSpp::LBFGSParam<double> param;
    param.epsilon = 1.0e-3;
    // param.past = 5;
    //param.delta = 1.0E-6;
    param.max_iterations = 25;
    LBFGSpp::LBFGSSolver<double> solver(param);
    GaussianProcessOpt fun;    
    fun.parent( this );

    // compute the baseline with which other evaluations are normalized
    log_likelihood_baseline = log_likelihood();

    Eigen::VectorXd x(covf().get_param_dim());
    x.setZero();
    double fx;
    int niter = solver.minimize( fun , x , fx );
    cf->loghyper_changed = true;

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
  }
}
