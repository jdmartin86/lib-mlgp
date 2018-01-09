/**
 * test_hetero_gp_regression
 *
 * This test evaluates the most likely heteroscedastic GP to
 * regress a sample from the f-process posterior.
 *
 * John Martin Jr. ( jmarti3@stevens.edu ) 
 */
#include "gp.h"
#include "gp_utils.h"

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

#define SQ( x ) ( ( x ) * ( x ) )

/**
 * linspace
 *
 * computes a uniform discretization over a line
 */
Eigen::VectorXd linspace( double m , double M , size_t N )
{
  double step = ( M - m ) / ( N - 1 );
  Eigen::VectorXd X( N );
  X[0] = m;
  for( size_t i = 1 ; i < N ; i++ ) X[i] = X[i-1] + step;
  return X;
}

/**
 * williams_data
 * 
 */
void williams_data( Eigen::VectorXd& x , Eigen::VectorXd& y )
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist( 0.0 , 1.0 );
  std::normal_distribution<double> z( 0.0 , 1.0 );

  for( size_t i = 1 ; i < x.size() ; i++ ) 
  {
    x(i) = dist(gen);
    double m = 2.0*sin(2.0*M_PI*x(i));
    double s = 0.5 + x(i); 
    y(i) =  m + s*z(gen);
  }
}

/**
 * test_mlgp_regression
 *
 * Test MLGP regression on synthetic data, originally used by Williams (1996).
 */
double test_mlgp_regression(libgp::GaussianProcess * gp)
{
  int input_dim = gp->get_input_dim(); // should be one
  size_t n = 200;

  Eigen::VectorXd x_tr( n );
  Eigen::VectorXd y( n );
  williams_data( x_tr , y );

  for(size_t i = 0; i < n*0.8; ++i) 
  {
    double x[input_dim];
    x[0] = x_tr[i];
    gp->add_pattern(x, y(i));
  }

  // test loop
  double tss = 0;
  for(size_t i = n*0.8+1; i < n; ++i) 
  {
    double x[input_dim];
    x[0] = x_tr[i];
    double f = gp->mean(x);
    double error = f - y(i);
    tss += error*error;
  }

  std::ofstream tfile;
  tfile.open( "test.txt" );
  tfile << "x,f,s,g,y" << std::endl;
  size_t N = 500;
  Eigen::VectorXd x_ts = linspace( 0.0 , 1.0 , N );
  for(size_t i = 0; i < N; ++i) 
  {
    double x[input_dim];
    x[0] = x_ts[i];
    double f = gp->mean(x);
    double s = gp->var(x);
    double g = gp->g(x);
    tfile << x[0] << "," << f << "," 
	  << s << "," << g << "," << 2.0*sin(2.0*M_PI*x[0])<< std::endl;
  }
  tfile.close();

  // output training file after noise targets have been updated
  std::ofstream ofile;
  ofile.open( "train.txt" );
  ofile << "x,y,z" << std::endl;
  for(size_t i = 0; i < n*0.8; ++i) 
  {
    ofile << x_tr[i] << "," << y(i) << "," << gp->get_z(i) << std::endl;
  }
  ofile.close();


  return tss/(n*0.2-1);
}

void run_regression_test( std::string covf_str,
			  std::string covh_str )
{
  double mss = 0.0;
  int n=1;
  for (int i=0; i<n; ++i) {
    int input_dim = 1;//libgp::Utils::randi(2) + 2;
    libgp::GaussianProcess* gp = new libgp::GaussianProcess(input_dim,
							    covf_str,
							    covh_str );
    Eigen::VectorXd fparams(gp->covf().get_param_dim());
    fparams.setZero();
    fparams(1) = log(0.22);
    fparams(gp->covf().get_param_dim()-1) = log(1.0E-6);
    gp->covf().set_loghyper(fparams);

    Eigen::VectorXd hparams(gp->covh().get_param_dim());
    hparams.setZero();
    hparams(0) = log( 1.0 );
    hparams(gp->covh().get_param_dim()-1) = log(0.5);
    gp->covh().set_loghyper(hparams);
    
    mss += test_mlgp_regression(gp);    
    delete gp;
  }
  ASSERT_TRUE(mss/n < 0.05);
}

TEST(HeteroGPRegressionTest, SEiso)
{
  std::string covf_str("CovSum( CovSEiso, CovNoise )");
  std::string covh_str("CovSum( CovSEiso, CovNoise )");
  run_regression_test( covf_str , covh_str );
}
/*
TEST(HeteroGPRegressionTest, Matern3iso) {
  std::string covf_str("CovSum ( CovMatern3iso, CovNoise)");
  std::string covh_str("CovSum ( CovMatern3iso, CovNoise)");
  run_regression_test(covf_str,covh_str);
}

TEST(HeteroGPRegressionTest, Matern5iso) {
  std::string covf_str("CovSum ( CovMatern5iso, CovNoise)");
  std::string covh_str("CovSum ( CovMatern5iso, CovNoise)");
  run_regression_test(covf_str,covh_str);
}

TEST(HeteroGPRegressionTest, CovSEard) {
  std::string covf_str("CovSum ( CovSEard, CovNoise)");
  std::string covh_str("CovSum ( CovSEard, CovNoise)");
  run_regression_test(covf_str,covh_str);
}

TEST(HeteroGPRegressionTest, CovRQiso) {
  std::string covf_str("CovSum ( CovRQiso, CovNoise)");
  std::string covh_str("CovSum ( CovRQiso, CovNoise)");
  run_regression_test(covf_str,covh_str);
}
*/
TEST(HeteroGPRegressionTest, UpdateL) {
  int input_dim = 1;
  libgp::GaussianProcess* gp 
    = new libgp::GaussianProcess(input_dim,
				 "CovSEiso", 
				 "CovSum( CovSEiso, CovNoise)" );

   Eigen::VectorXd fparams(gp->covf().get_param_dim());
   fparams.setZero();
   // fparams(gp->covf().get_param_dim()-1) = -2;
   gp->covf().set_loghyper(fparams);

   Eigen::VectorXd hparams(gp->covh().get_param_dim());
   hparams.setZero();
   hparams(0) = -10;
   hparams(gp->covh().get_param_dim()-1) = -10;
   gp->covh().set_loghyper(hparams);
   
  size_t n = 100;
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd z(n);
  Eigen::VectorXd y = gp->draw_random_hetero_sample(X,z);

  std::ofstream ofile;
  ofile.open ("fsample.txt");
  ofile << "x,y,z\n";
  for(size_t i = 0; i < n; ++i) {
    double x[input_dim];
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp->add_pattern(x, y(i));
    
    ofile << x[0] << "," 
	  << y(i) << ","
	  << z(i) << std::endl;

  }
  Eigen::VectorXd x(input_dim);
  x.setZero();
  gp->f(x);
  ofile.close();
}
