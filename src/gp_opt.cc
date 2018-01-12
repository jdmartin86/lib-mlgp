/** 
 * gp_opt.cc
 *
 * Optimizer interface for GP class
 *
 * John Martin Jr. ( jmarti3@stevens.edu )
 */
#include "gp_opt.h"
namespace libgp 
{
  double GaussianProcessOpt::value( const Eigen::VectorXd& x )
  {
    gp->covf().set_loghyper(x);
    return gp->norm_log_likelihood();
  }

  void GaussianProcessOpt::gradient( const Eigen::VectorXd& x , 
				     Eigen::VectorXd& grad )
  {
    gp->covf().set_loghyper(x);   
    grad = gp->log_likelihood_gradient();
  }

  double GaussianProcessOpt::operator()( const Eigen::VectorXd& x, 
					 Eigen::VectorXd& grad )
  {
    gradient( x , grad );
    return value( x );
  }

  double GaussianProcessfOpt::value( const Eigen::VectorXd& x )
  {
    gp->covf().set_loghyper(x);
    return gp->norm_log_flikelihood();
  }

  void GaussianProcessfOpt::gradient( const Eigen::VectorXd& x , 
				      Eigen::VectorXd& grad )
  {
    gp->covf().set_loghyper(x);   
    grad = gp->log_flikelihood_gradient();
  }

  double GaussianProcessfOpt::operator()( const Eigen::VectorXd& x, 
					 Eigen::VectorXd& grad )
  {
    gradient( x , grad );
    return value( x );
  }
}
