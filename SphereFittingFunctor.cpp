#include "SphereFitting.h"

#include "SphereFittingFunctor.h"

#include <algorithm>
#include <cmath>

namespace MatchingTools
{  
  SphereFittingFunctor::SphereFittingFunctor(const Matrix3Xf& i_points) :
    LMFunctor<float>(4, i_points.cols()),  points(i_points)
  {}

  int SphereFittingFunctor::operator()(const VectorXf &i_pVals, VectorXf &o_fRes) const
  {
    assert(o_fRes.rows() == points.cols());
    
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
	
    const auto kappa = i_pVals(0);    
    const auto rho = i_pVals(1);
    const auto phi = i_pVals(2);
    const auto theta = i_pVals(3);

    const auto sint = std::sin(theta);
    const auto wVec = Eigen::Vector3f(std::cos(phi) * sint, std::sin(phi) * sint, std::cos(theta));
    for (Eigen::Index i = 0; i < o_fRes.rows(); ++i)
    {
       const Eigen::Vector3f qVec = points.col(i) - (rho * wVec);
       const auto qqDot = qVec.dot(qVec);
       const auto qwDot = qVec.dot(wVec);
       o_fRes(i) = 0.5 * kappa * qqDot - qwDot;
    }
	
    return 0;
  }

  int SphereFittingFunctor::df(const VectorXf &i_pVals, MatrixXf &o_fJac) const
  {
    assert(o_fJac.rows() == points.cols());
    assert(o_fJac.cols() == i_pVals.rows());
    
    const auto kappa = i_pVals(0);    
    const auto rho = i_pVals(1);
    const auto phi = i_pVals(2);
    const auto theta = i_pVals(3);

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

    for (Eigen::Index i = 0; i < o_fJac.rows(); ++i)
    {
       Eigen::Vector4f gradVec;
	     
       const Eigen::Vector3f qVec = points.col(i) - (rho * wVec);
       
       // Evaluate 1st-order partial derivative with respect to radius.
       gradVec(0) = qVec.dot(qVec);

       // Evaluate 1st-order partial derivative with respect to rho.
       gradVec(1) = 1.f - kappa * qVec.dot(wVec);

       // Evaluate 1st-order partial derivative with respect to phi.
       gradVec(2) = kappa * qVec.dot(dqVecdPhi) - (dotPhi + qVec.dot(dwVecdPhi));

       // Evaluate 1st-order partial derivative with respect to theta.
       gradVec(3) = kappa * qVec.dot(dqVecdTheta) - (dotTheta + qVec.dot(dwVecdTheta));

       o_fJac.block(i, 0, 1, 4) = gradVec.transpose();
    }

    return 0;
  }
  
}
