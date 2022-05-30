#pragma once

/** 
 * \file SphereFitting.hpp
 *
 * \brief Definition of class \c SphereFitting.
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
   enum class SphereFittingAlgorithm
   {
     Algebraic,
     LinearGeometric,
     NonLinearGeometric
   };
  
  /** 
   * \class SphereFitting
   *
   * \brief This class implements three algorithms to perform the same
   *        task: best fit a sphere to a given set of points in space.
   *
   * \note One algorithm is based on the algebraic approach, while the
   *       other  two  are  based  on  the  geometric  approach.   The
   *       algebraic  one  is  simpler  and faster,  but  it  produces
   *       slightly less accurate results (even  if the noise level is
   *       negligible).   Geometric approaches  are  supposedly to  be
   *       more  accurate,  but  they   are  more  complex  too.   One
   *       geometric algorithm is based on minimizing a sum of squares
   *       distances  whose  residuals  are  affine.  So,  it  can  be
   *       minimized  by  least-squares.   We   named  it  the  linear
   *       geometric   algorithm.   The   other  geometric   algorithm
   *       minimizes  a  sum  of   squares  distances  whose  residual
   *       functions is non-linear.  So, it  is minimized by solving a
   *       nonlinear  least-squares  problem,  which  should  be  more
   *       accurate than two previous algorithms,  but also a lot more
   *       expensive.   This nonlinear  geometric  algorithm needs  an
   *       initial  guess. In  the class  implementation, the  initial
   *       guess  is always  the  solution provided  by the  algebraic
   *       algorithm.
   */

  class SphereFitting
  {
  public:
    using Matrix3Xf = Eigen::Matrix3Xf;
    using Vector3f = Eigen::Vector3f;
  
    /**
     * \struct  CenterRadiusPair 
     *
     * \brief Type representing the center and radius of a sphere.
     */
    struct CenterRadiusPair
    {
      Vector3f center;  ///< Center of the best fitted sphere
      double radius;    ///< Radius of the best fitted sphere
    };

    /**
     * \fn SphereFitting()
     *
     * \brief Constructor.
     * \param alg A choice for the best sphere fit algorithm.
     */
    explicit SphereFitting(SphereFittingAlgorithm alg = SphereFittingAlgorithm::Algebraic);

    /**
     * \fn run()  
     *
     * \brief Computes  center and radius  of a sphere that  best fits
     * the given set of points according  to the best fit criterion of
     * the chosen algorithm.
     * \param points Coordinates of a set of points in space.
     * \return Center and radius of a sphere that best fits the points
     * according to the best fit criterion of the chosen algorithm.
     */
    CenterRadiusPair run(const Matrix3Xf& points) const;

  private:
    using Matrix3f = Eigen::Matrix3f;
    
    /** Computes a best sphere fitting using an algebraic approach. */
    CenterRadiusPair fitSphereUsingAlgebraicApproach(const Matrix3Xf& points) const;

    /** Computes a best sphere fitting using a linear geometric approach. */
    CenterRadiusPair fitSphereUsingLinearGeometricApproach(const Matrix3Xf& points) const;

    /** Computes a best sphere fitting using a nonlinear geometric approach. */
    CenterRadiusPair fitSphereUsingNonLinearGeometricApproach(const Matrix3Xf& points) const;

  private:

    SphereFittingAlgorithm _alg; ///< Choice of best fit sphere algorithm.
    
  };
  
}
