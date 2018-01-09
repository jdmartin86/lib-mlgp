/** 
 * gp_opt.h
 *
 * Optimizer interface for GP class
 *
 * John Martin Jr. ( jmarti3@stevens.edu )
 */
#ifndef GP_OPT_H
#define GP_OPT_H

#include "gp.h"

#include <Eigen/Core>
namespace libgp 
{
  class GaussianProcessOpt
  {
    GaussianProcess* gp;

  public:
    GaussianProcessOpt(){};
    virtual ~GaussianProcessOpt(){};
    
    void parent( GaussianProcess* g ){ gp = g; };

    double value( const Eigen::VectorXd& x );

    void gradient( const Eigen::VectorXd& x , Eigen::VectorXd& grad );

    double operator()( const Eigen::VectorXd& x , Eigen::VectorXd& grad );
  };

  class GaussianProcessfOpt
  {
    GaussianProcess* gp;

  public:
    GaussianProcessfOpt(){};
    virtual ~GaussianProcessfOpt(){};
    
    void parent( GaussianProcess* g ){ gp = g; };

    double value( const Eigen::VectorXd& x );

    void gradient( const Eigen::VectorXd& x , Eigen::VectorXd& grad );

    double operator()( const Eigen::VectorXd& x , Eigen::VectorXd& grad );
  };
}

#endif
