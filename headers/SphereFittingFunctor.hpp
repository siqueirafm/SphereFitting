#pragma once

/** 
 * \file SphereFittingFunctor.hpp
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

#include <Eigen/Core>

namespace MatchingTools
{
  /*
   * A generic functor to use Eigen's Levenberg-Marquardt class.
   */
  template<typename S, int N = Eigen::Dynamic, int M = Eigen::Dynamic>
  struct LMFunctor
  {
    using Scalar = S;

    enum {
      InputsAtCompileTime = N,
      ValuesAtCompileTime = M
    };
    
    using InputType = Eigen::Matrix<Scalar, InputsAtCompileTime, 1>;
    using ValueType = Eigen::Matrix<Scalar, ValuesAtCompileTime, 1>;
    using JacobianType = Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>;
    
    Eigen::Index numberOfInputs;  ///< This is equivalent to the dimension of the range of the residual function
    Eigen::Index numberOfValues;  ///< This is equivalent to the dimension of the domain of the residual function

    LMFunctor() : numberOfInputs(InputsAtCompileTime), numberOfValues(ValuesAtCompileTime)
    {}
    
    LMFunctor(Eigen::Index n, Eigen::Index m) : numberOfInputs(n), numberOfValues(m)
    {}

    Eigen::Index inputs() const { return numberOfInputs; }
    Eigen::Index values() const { return numberOfValues; }
  };

  /**
   * The Levenberg-Marquardt (LM)  solver from Eigen must  be given an
   * implementation of  the residual  function and of  its first-order
   * differential.  This struct  defines a  functor with  the required
   * information.
   */
  struct SphereFittingFunctor : LMFunctor<double>
  {
    using Matrix3Xd = Eigen::Matrix3Xd;
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
      
    Matrix3Xd _points;  ///< Set of points the sphere is supposed to fit.

    explicit SphereFittingFunctor(const Matrix3Xd& points);
      
    /*
     * Evaluates the  residual function f :  R^n --> R^m at  a given
     * set of parameter values, where  n is the number of parameters
     * of the model, and m is the number of points.
     */
    int operator()(const VectorXd &pVals, VectorXd &fRes) const;

    /*
     * Evaluates the 1st-order differential df  : R^n --> R^m of the
     * the function  f :  R^n -->  R^m at a  given set  of parameter
     * values, where n is the number of parameters of the model, and
     * m is the number of points.
     */
    int df(const VectorXd &pVals, MatrixXd &fJac) const;
  };

}
