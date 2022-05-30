/** 
 * \file SphereFittingFunctor.cpp
 *
 * \brief Implementation of class \c SphereFitting.
 *
 * \author
 * Marcelo Ferreira Siqueira \n
 * mfsiqueira at gmail (dot) com
 *
 * \version 1.0
 * \date November 2020
 *
 * \attention This program is distributed WITHOUT ANY WARRANTY, and it
 *            may be freely redistributed under the condition that the
 *            copyright notices  are not removed,  and no compensation
 *            is received. Private, research, and institutional use is
 *            free. Distribution of this  code as part of a commercial
 *            system  is permissible ONLY  BY DIRECT  ARRANGEMENT WITH
 *            THE AUTHOR.
 */

#include <SphereFitting.hpp>

#include <SphereFittingFunctor.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <unsupported/Eigen/NonLinearOptimization>

#include <cmath>
#include <stdexcept>

namespace MatchingTools
{
  namespace
  {
    // A singular value is considered zero  if its value is no greater
    // than  threshold times  the largest  singular value  in absolute
    // value.
    constexpr auto ZeroSingularValueThreshold = 0.00001f;

    Eigen::VectorXf computeParameterValuesFromCenterAndRadius(const SphereFitting::CenterRadiusPair& centerRadius)
    {
      const static auto MyPi = acos(-1.f);

      Eigen::VectorXf values(4);

      const auto centerVecNorm = centerRadius.center.norm();
      assert(centerVecNorm > 0.f);
      const auto unitCenterVec = centerRadius.center * (1.f / centerVecNorm);
      
      assert(centerRadius.radius > 0.f);
      values(0) = 1.f / centerRadius.radius;
      values(1) = centerVecNorm - centerRadius.radius;
   
      const auto lengthProjXY = unitCenterVec.head(2).norm();
      if (lengthProjXY == 0.f)
        values(2) = 0.f;
      else if (unitCenterVec(0) >= 0.f)
      {
        values(2) = std::asin(unitCenterVec(1) / lengthProjXY) + (unitCenterVec(1) >= 0.f ? 0 : 2.f * MyPi);
      }
      else
        values(2) = MyPi - std::asin(unitCenterVec(1) / lengthProjXY);
    
      values(3) = std::acos(std::clamp(unitCenterVec(2), -1.f, 1.f));

      return values;      
    }

    SphereFitting::CenterRadiusPair computeRadiusAndCenterFromParameterValues(const Eigen::Vector4f& params)
    {
      assert(params(0) > 0.f);
      
      const auto radius = 1.f / params(0);

      const auto rho = params(1);
      const auto phi = params(2);
      const auto theta = params(3);
      
      const auto sint = std::sin(theta);
      const auto wVec = Eigen::Vector3f(std::cos(phi) * sint, std::sin(phi) * sint, std::cos(theta));

      const auto center = (radius + rho) * wVec;

      return { center, radius };
    }
  }
  
  SphereFitting::SphereFitting(SphereFittingAlgorithm alg) : _alg(alg)
  {
  }

  SphereFitting::CenterRadiusPair SphereFitting::run(const Matrix3Xf& points) const
  {
    if (points.cols() < 4)
      throw std::runtime_error("Number of points should be at least 4");

    switch (_alg)
    {
    case SphereFittingAlgorithm::Algebraic:
      return fitSphereUsingAlgebraicApproach(points);
    case SphereFittingAlgorithm::LinearGeometric:
      return fitSphereUsingLinearGeometricApproach(points);
    case SphereFittingAlgorithm::NonLinearGeometric:
      return fitSphereUsingNonLinearGeometricApproach(points);
    default:
      throw std::runtime_error("Unknown choice of best fit algorithm");
    }
    
    return {};
  }

  ///////////////////////////////////////////////////////////////////////////////
  ///
  /// Algorithm from the paper:
  ///
  /// Samuel M. Thomas and Y. T. Chan                      
  /// A simple approach for the estimation of circular arc center and radius,
  /// Computer Vision, Graphics, and Image Processing, 45, p. 362-370, 1989.
  ///
  /// Rationale:
  ///
  /// Minimize a functional that takes into account the sphere surface area:
  ///
  /// F(r,x_c,y_c,z_c) = \sum_{i = 1}^n ( 4\pi r^2 - 4\pi( (x_i - x_c)^2 + (y_i - y_c)^2 + (z_i - z_c)^2 ) )^2
  ///
  /// The equations grad F = 0 can be combined and simplified to yield a linear
  /// system with three equations and three unknowns (i.e., x_c, y_c, and z_c).
  /// The solution of this system is then used to compute the remaining unknown
  /// r.
  ///
  SphereFitting::CenterRadiusPair SphereFitting::fitSphereUsingAlgebraicApproach(const Matrix3Xf& points) const
  {
    const auto numberOfPoints = points.cols();

    // Square each element of the 3 x N matrix of point coordinates. 
    // 
    //             | -------- x^2 -------- |
    // pointsSqr = | -------- y^2 -------- |
    //             | -------- z^2 -------- |
    //
    // Here, a^2 denotes the result of squaring all elements of row a.
    
    const Matrix3Xf pointsSqr = points.cwiseProduct(points);


    // Computes the inner  product of every two rows of  the matrix of
    // point coordinates.
    //
    //                 | x.x  x.y  x.z |         
    // rowColProdOne = | y.x  y.y  y.z |
    //                 | z.x  z.y  z.z |
    //
    // Here a.b denotes the inner product of rows a and b.

    const Matrix3f rowColProdOne = points * points.transpose();

    
    // Computes a 3x3 matrix from the previous two:
    //
    //                 | x.x^2  x.y^2  x.z^2 |         
    // rowColProdTwo = | y.x^2  y.y^2  y.z^2 |
    //                 | z.x^2  z.y^2  z.z^2 |
    //
    // Here  a.b^2  denotes  the  inner  product  of  row  a  and  the
    // element-wise squared row b.

    const Matrix3f rowColProdTwo = points * pointsSqr.transpose();

    
    // Computes a 3x1 vector such that  the i-th element is the sum of
    // the i-th coordinates of the given points.
    //
    //             | sum_x |
    // rowSumOne = | sum_y |
    //             | sum_z |
    //
    // Here, sum_a denotes the sum of all a-coordinates of the points.

    const Vector3f rowSumOne = points.rowwise().sum();

    
    // Computes a 3x1 vector from the sum of the rows of matrix rowColProdTwo:
    //
    //             | x.x^2 + x.y^2 + x.z^2 |
    // rowSumTwo = | y.x^2 + y.y^2 + y.z^2 |
    //             | z.x^2 + z.y^2 + z.z^2 |

    const Vector3f rowSumTwo = rowColProdTwo.rowwise().sum();

    
    // Computes the outer product of vector rowSumOne with itself:
    //
    //                           | sum_x * sum_x  sum_x * sum_y  sum_x * sum_z |
    // rowSumOne ^ rowSumOne^t = | sum_y * sum_x  sum_y * sum_y  sum_y * sum_z |
    //                           | sum_z * sum_x  sum_z * sum_y  sum_z * sum_z |
    //
    // Here, sum_a denotes the sum of all a-coordinates of the points.

    const Matrix3f rowSumOneOuterProd = rowSumOne * rowSumOne.transpose();


    // Create matrix A and vector b
    //
    //         | (sum_x * sum_x - N * x.x)  (sum_x * sum_y - N * x.y)  (sum_x * sum_z - N * x.z) |
    // A = 2 * | (sum_y * sum_x - N * y.x)  (sum_y * sum_y - N * y.y)  (sum_y * sum_z - N * y.z) |
    //         | (sum_z * sum_x - N * z.x)  (sum_z * sum_y - N * z.y)  (sum_z * sum_z - N * z.z) |
    //
    // and
    //
    //         | -N * ( x.x^2 + x.y^2 + x.z^2 ) + S * sum_x |
    // b =     | -N * ( y.x^2 + y.y^2 + y.z^2 ) + S * sum_y |
    //         | -N * ( z.x^2 + z.y^2 + z.z^2 ) + S * sum_z |
    //
    // where
    //
    // S is the sum of the main diagonal of matrix 'rowColProdOne'.

    const float scaleFactor = rowColProdOne.diagonal().sum();
    const Matrix3f matA = 2 * (rowSumOneOuterProd - numberOfPoints * rowColProdOne);
    const Vector3f vecB = -numberOfPoints * rowSumTwo + scaleFactor * rowSumOne;

    //
    // Solve A x = b
    //  
    const Vector3f center = matA.colPivHouseholderQr().solve(vecB);

 
    //
    // Compute the radius of the sphere using the center coordinates.
    //
    const float ctc = center.transpose() * center;
    const float cts = center.transpose() * rowSumOne;

    const auto radiusSqr = ctc + (scaleFactor - 2.f * cts) / numberOfPoints;
    assert(radiusSqr >= 0.f);

    return {center, std::sqrt(radiusSqr) };
  }

  ///////////////////////////////////////////////////////////////////////////////
  ///
  /// Algorithm from the paper:
  ///
  /// I. D. Coope
  /// Circle Fitting by Linear and Nonlinear Least Squares
  /// Journal of Optimization Theory and Applications, 76(2), p. 381-388, 1993.
  ///
  /// Rationale:
  ///
  /// Minimize a functional that takes into account geometric distance: 
  ///
  /// F(r,x_c,y_c,z_c) = \sum_{i = 1}^n f_i(r,x_c,y_c,z_c)^2
  ///
  /// where
  ///
  /// f_i(r,x_c,y_c,z_c) = || (x_i, y_i, z_i) - (x_c, y_c, z_c) ||^2 - r^2
  ///
  /// The residual function is nonlinear, but it can be cleverly re-written to
  /// remove the non-linearity through a change of coordinates. Indeed, we have
  /// that
  ///
  /// f_i(c, r) = c^tc - 2c^t p_i + p_i^t p_i - r^2
  ///
  /// where
  ///
  /// c = (x_c, y_c, z_c) and p_i = (x_i, y_i, z_i).
  ///
  /// By defining
  ///
  /// q = (2 x_c, 2 y_c, 2 z_c, r^2 - c^tc ) in R^4
  ///
  /// and
  ///
  /// b_j = [ p_j  1 ]^t = [ x_j  y_j  z_j  1 ]^t ,      j = 1,...,n
  ///
  /// we get
  ///
  /// g_j( q ) = p_j^t p_j - b_j^j q ,                   j = 1,...,n
  ///
  /// which is affine. So, we can minimize the following functional using least-
  /// squares:
  ///
  /// G( q ) = \sum_{i = 1}^n g_i( q )^2
  ///
  /// and then we compute center and radius as follows:
  ///
  /// x_c = q_1 / 2, y_c = q_2 / 2, z_c = q_3 / 2, and r = sqrt( q_4 + c^t c ).
  ///
  
  SphereFitting::CenterRadiusPair SphereFitting::fitSphereUsingLinearGeometricApproach(const Matrix3Xf& points) const
  {
    Eigen::Matrix4Xf matB(4, points.cols());
    Eigen::VectorXf vecD(points.cols());

    matB.topLeftCorner(3, points.cols()) = points;
    matB.row(3).setOnes();
   
    vecD = points.cwiseProduct(points).colwise().sum().transpose();

    Eigen::JacobiSVD<Eigen::MatrixX4f> svd(matB.transpose(), Eigen::ComputeFullV | Eigen::ComputeFullU);
    
    svd.setThreshold(ZeroSingularValueThreshold);
    assert(svd.rank() == 4);

    Eigen::Vector4f vecY = svd.solve(vecD);

    SphereFitting::CenterRadiusPair p;
    p.center = 0.5f * vecY.head(3);
    p.radius = std::sqrt(vecY(3) + p.center.transpose() * p.center);

    return p;
  }


  ///////////////////////////////////////////////////////////////////////////////
  ///
  /// Algorithm from the paper:
  ///
  /// Gabor Lukács, Ralph Martin, and Dave Marshall.
  /// Faithful Least-SquaresFitting  of Spheres, Cylinders,  Cones and
  /// Tori  for Reliable  Segmentation,  Proceedings  of the  ECCV'98,
  /// Lecture Notes  in Computer  Science, v. 1406,  Springer, Berlin,
  /// Heidelberg, 1998.
  /// 
  /// Rationale:
  ///
  /// Solve a non-linear least-squares problem in which the functional
  /// is a sum  of squares of a non-linear  function that approximates
  /// the Euclidean distance of a point to the sphere. The approximate
  /// function meets  some properties  that makes  it faithful  to the
  /// true distance, yet easier, more  stable, and more robust when it
  /// comes to the numerical  optimization process, which is conducted
  /// by the  Levenberg-Marquardt (LM) algorithm implemented  in Eigen
  /// (unsupported module).
  ///

  SphereFitting::CenterRadiusPair SphereFitting::fitSphereUsingNonLinearGeometricApproach(const Matrix3Xf& points) const
  {
    // Find initial solution using the algebraic approach.
    auto p = fitSphereUsingAlgebraicApproach(points);
    auto params = computeParameterValuesFromCenterAndRadius(p);

    // Improve  upon initial  solution by  an iterative,  minimization
    // process  conducted  by  the  Levenberg-Marquardt  algorithm  of
    // Eigen.
    SphereFittingFunctor spFunctor(points);
    Eigen::LevenbergMarquardt<SphereFittingFunctor, SphereFittingFunctor::Scalar> solver(spFunctor);

    // Compute the norm of the residual vector before optimization.
    Eigen::VectorXf fResBefore(points.cols());
    spFunctor(params, fResBefore);
    const auto normBefore = fResBefore.norm();

    // Maximum number of function evaluations.
    solver.parameters.maxfev = 500;
    const auto status = solver.minimize(params);
    
    switch(status)
    {
    case Eigen::LevenbergMarquardtSpace::RelativeReductionTooSmall:
    case Eigen::LevenbergMarquardtSpace::RelativeErrorTooSmall:
    case Eigen::LevenbergMarquardtSpace::RelativeErrorAndReductionTooSmall:
    case Eigen::LevenbergMarquardtSpace::CosinusTooSmall:
    case Eigen::LevenbergMarquardtSpace::TooManyFunctionEvaluation:
    case Eigen::LevenbergMarquardtSpace::FtolTooSmall:
    case Eigen::LevenbergMarquardtSpace::XtolTooSmall:
    case Eigen::LevenbergMarquardtSpace::GtolTooSmall:
      {
	Eigen::VectorXf fResAfter(points.cols());
        spFunctor(params, fResAfter);
        const auto normAfter = fResAfter.norm();
	
	// Update solution if and only if the norm of the residual vector decreased.
	if (normAfter < normBefore)
	  p = computeRadiusAndCenterFromParameterValues(params);
      }
      break;
    case Eigen::LevenbergMarquardtSpace::ImproperInputParameters:
      throw std::runtime_error("Check consistency of input parameters passed to the LM solver");
    default:
      throw std::runtime_error("Unexpected LM solver termination status");
    }

    return p;
  }  
}
