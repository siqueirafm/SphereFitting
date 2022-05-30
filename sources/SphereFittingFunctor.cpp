/** 
 * \file SphereFittingFunctor.cpp
 *
 * \brief A functor for using the Eigen's Levenberg-Marquardt class.
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

#include <algorithm>
#include <cmath>

namespace MatchingTools
{  
  SphereFittingFunctor::SphereFittingFunctor(const Matrix3Xf& points) :
    LMFunctor<float>(4, points.cols()),  _points(points)
  {}

  int SphereFittingFunctor::operator()(const VectorXf &pVals, VectorXf &fRes) const
  {
    assert(fRes.rows() == _points.cols());
    
    // Each member function  of the residual function f :  R^n --> R^m
    // implemented below is an approximation to the Euclidean distance
    // from a  point in space to  a sphere. There are  four parameters
    // (i.e., n = 4):
    //
    // 1) kappa
    // 2) rho
    // 3) phi
    // 4) theta
    //
    // where  'kappa' is  the  inverse  of the  radius  of the  sought
    // sphere,  'rho' is  the distance  of  the closest  point of  the
    // sphere to the origin O of  the coordinate system, and 'phi' and
    // 'theta' are the spherical coordinates of a unit vector 'w' such
    // that 'O  + rho  w' is the  closest point of  the sphere  to the
    // origin.
    //
    // Each coordinate-function f_i : R^4 --> R is defined as
    //
    // f_i( radius, rho, phi, theta ) = (kappa / 2 ) < q, q > - < q, w >
    //
    // where
    //
    // q = pt(i) - rho w
    //
    // and
    //
    // pt(i)
    //
    // is the i-th point of the set to which the sphere is fitted.
	
    const auto kappa = pVals(0);    
    const auto rho = pVals(1);
    const auto phi = pVals(2);
    const auto theta = pVals(3);

    const auto sint = std::sin(theta);
    const auto wVec = Eigen::Vector3f(std::cos(phi) * sint, std::sin(phi) * sint, std::cos(theta));
    for (Eigen::Index i = 0; i < fRes.rows(); ++i)
    {
       const Eigen::Vector3f qVec = _points.col(i) - (rho * wVec);
       const auto qqDot = qVec.dot(qVec);
       const auto qwDot = qVec.dot(wVec);
       fRes(i) = 0.5 * kappa * qqDot - qwDot;
    }
	
    return 0;
  }

  int SphereFittingFunctor::df(const VectorXf &pVals, MatrixXf &fJac) const
  {
    assert(fJac.rows() == _points.cols());
    assert(fJac.cols() == pVals.rows());
    
    const auto kappa = pVals(0);    
    const auto rho = pVals(1);
    const auto phi = pVals(2);
    const auto theta = pVals(3);

    const auto sinp = std::sin(phi);
    const auto cosp = std::cos(phi);
    const auto sint = std::sin(theta);
    const auto cost = std::cos(theta);

    const auto wVec = Eigen::Vector3f(cosp * sint, sinp * sint, cost);

    const auto dwVecdPhi = Eigen::Vector3f(-sinp * sint, cosp * sint, 0.f);
    const auto dqVecdPhi = -rho * dwVecdPhi;
    const auto dotPhi = dqVecdPhi.dot(wVec);

    const auto dwVecdTheta = Eigen::Vector3f(cosp * cost, sinp * cost, -sint);
    const auto dqVecdTheta = -rho * dwVecdTheta;
    const auto dotTheta = dqVecdTheta.dot(wVec);

    for (Eigen::Index i = 0; i < fJac.rows(); ++i)
    {
       Eigen::Vector4f gradVec;
	     
       const Eigen::Vector3f qVec = _points.col(i) - (rho * wVec);
       
       // Evaluate 1st-order partial derivative with respect to radius.
       gradVec(0) = qVec.dot(qVec);

       // Evaluate 1st-order partial derivative with respect to rho.
       gradVec(1) = 1.f - kappa * qVec.dot(wVec);

       // Evaluate 1st-order partial derivative with respect to phi.
       gradVec(2) = kappa * qVec.dot(dqVecdPhi) - (dotPhi + qVec.dot(dwVecdPhi));

       // Evaluate 1st-order partial derivative with respect to theta.
       gradVec(3) = kappa * qVec.dot(dqVecdTheta) - (dotTheta + qVec.dot(dwVecdTheta));

       fJac.block(i, 0, 1, 4) = gradVec.transpose();
    }

    return 0;
  }
  
}
