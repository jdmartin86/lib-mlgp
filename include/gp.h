// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*! 
 *  
 *   \page licence Licensing
 *    
 *     libgp - Gaussian process library for Machine Learning
 *
 *      \verbinclude "../COPYING"
 */

#ifndef __GP_H__
#define __GP_H__

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

#include "cov.h"
#include "sampleset.h"
#include "gp_utils.h"

namespace libgp {
  
  /** Gaussian process regression.
   *  @author Manuel Blum */
  class GaussianProcess
  {
  public:

    /** Empty initialization */
    GaussianProcess ();
    
    /** Create and instance of GaussianProcess with given input dimensionality 
     *  and covariance function. */
    GaussianProcess (size_t input_dim, std::string covf_def);

    /** Create and instance of GaussianProcess with given input dimensionality 
     *  and covariance function. */
    GaussianProcess (size_t input_dim, 
		     std::string covf_def,
		     std::string covh_def );
    
    /** Create and instance of GaussianProcess from file. */
    GaussianProcess (const char * filename);
    
    /** Copy constructor */
    GaussianProcess (const GaussianProcess& gp);
    
    virtual ~GaussianProcess ();
    
    /** Write current gp model to file. */
    void write(const char * filename);
    
    /** Predict target value for given input.
     *  @param x input vector
     *  @return predicted value */
    virtual double f(const double x[]);
    virtual double f(const Eigen::VectorXd& x_star );

    /** Predict target value for given input.
     *  @param x input vector
     *  @return predicted value */
    virtual double g(const double x[]);
    virtual double g(const Eigen::VectorXd& x_star );

    /** Predict target value for given input.
     *  @param x input vector
     *  @return predicted value */
    virtual double mean(const double x[]);
    virtual double mean(const Eigen::VectorXd& x_star );
    
    /** Predict variance of prediction for given input.
     *  @param x input vector
     *  @return predicted variance */
    virtual double varf(const Eigen::VectorXd& x_star );
    virtual double varf(const double x[]);

    /** Predict variance of prediction for given input.
     *  @param x input vector
     *  @return predicted variance */
    virtual double varg(const Eigen::VectorXd& x_star );
    virtual double varg(const double x[]);

    /** Predict variance of prediction for given input.
     *  @param x input vector
     *  @return predicted variance */
    virtual double var(const Eigen::VectorXd& x_star );
    virtual double var(const double x[]);
    
    /** Add input-output-pair to sample set.
     *  Add a copy of the given input-output-pair to sample set.
     *  @param x input array
     *  @param y output value
     */
    void add_pattern(const double x[], double y);

    bool set_y(size_t i, double y);

    /** 
     * Get i-th h-process target
     */
    double get_z( size_t i );

    /** Get number of samples in the training set. */
    size_t get_sampleset_size();
    
    /** Clear sample set and free memory. */
    void clear_sampleset();
    
    /** Get reference of f-process covariance function. */
    CovarianceFunction & covf();

    /** Get reference of h-process covariance function. */
    CovarianceFunction & covh();

    /** Draw a random sample from the heteroscedastic prior */
    Eigen::VectorXd draw_random_hetero_sample( Eigen::MatrixXd& X,
					       Eigen::VectorXd& z);

    /** Get input vector dimensionality. */
    size_t get_input_dim();

    double log_flikelihood();
    double log_hlikelihood();
    double log_likelihood();
    
    Eigen::VectorXd log_flikelihood_gradient();
    Eigen::VectorXd log_hlikelihood_gradient();
    Eigen::VectorXd log_likelihood_gradient();

    /** Optimize hyperparams for f-process */
    void optimize_fhyper( );
    void optimize_hhyper( );
    void optimize_hyper( );

  protected:
    /** The h-process for heteroscedastic Gaussian processes */
    GaussianProcess* h;

    /** The covariance function of this Gaussian process. */
    CovarianceFunction * cf;
    
    /** The training sample set. */
    SampleSet * sampleset;
    
    /** Alpha is cached for performance. */ 
    Eigen::VectorXd alpha;
    Eigen::VectorXd alphaf;
    
    /** Last test kernel vector. */
    Eigen::VectorXd k_star;

    /** Linear solver used to invert the covariance matrix. */
    //    Eigen::LLT<Eigen::MatrixXd> solver;

    // Cholesky decomp for the composite process
    Eigen::MatrixXd L; 

    // Cholesky decomp for the homoscedastic process
    Eigen::MatrixXd Lf;
    
    /** Input vector dimensionality. */
    size_t input_dim;
    
    /** Update test input and cache kernel vector. */
    void update_k_star(const Eigen::VectorXd &x_star);

    void update_alpha();
    void update_alphaf();

     void update_noise( );

    /** Compute covariance matrix and perform cholesky decomposition. */
    virtual void compute();

    /** Homoscedastic process: Compute covariance matrix and
     *  perform cholesky decomposition.*/
    virtual void computef();
    
    bool alpha_needs_update;
    bool alphaf_needs_update;

    bool noise_needs_update;

  private:

    /** No assignement */
    GaussianProcess& operator=(const GaussianProcess&);

  };
}

#endif /* __GP_H__ */
