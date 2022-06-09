/** 
 * \file SphereFittingTest.cpp
 *
 * \brief A set of unit tests for the sphere fitting algorithms.
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

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <unsupported/Eigen/NonLinearOptimization>

using namespace MatchingTools;

namespace
{
  using Matrix3Xd = SphereFitting::Matrix3Xd;
  using CenterRadiusPair = SphereFitting::CenterRadiusPair;

  constexpr static auto ProximityTolerance = 0.00001;

  std::default_random_engine generator;

  Matrix3Xd generateWhiteNoise(int size, double stdev)
  {
    Matrix3Xd noise(3, size);

    std::normal_distribution<double> distrX(0.0, stdev);
    std::normal_distribution<double> distrY(0.0, stdev);
    std::normal_distribution<double> distrZ(0.0, stdev);
    
    for (Eigen::Index col = 0; col < noise.cols(); ++col)
      noise.col(col) << distrX(generator), distrY(generator), distrZ(generator);

    return noise;
  }

  // This simple method is described at
  //
  // Mervin  E.  Muller,  A Note  on  a Method  for Generating  Points
  // Uniformly  on N-Dimensional  Spheres Communications  of the  ACM,
  // 2(4), p. 19-20, 1959
  
  Matrix3Xd generateUniformlyDistributedPointsOnSphere(int size, const CenterRadiusPair& sphere)
  {
    Matrix3Xd samples(3, size);
    
    std::normal_distribution<double> distrX(0.0, 1.0);
    std::normal_distribution<double> distrY(0.0, 1.0);
    std::normal_distribution<double> distrZ(0.0, 1.0);
    
    for (Eigen::Index col = 0; col < samples.cols(); ++col)
    {
      samples.col(col) << distrX(generator), distrY(generator), distrZ(generator);
      samples.col(col) = sphere.radius * samples.col(col).normalized() + sphere.center;      
    }

    return samples;
  }

  Matrix3Xd generatePointsCloseToSphere(int size, const CenterRadiusPair& sphere, double stdev)
  {
    Matrix3Xd pointsOnSphere = generateUniformlyDistributedPointsOnSphere(size, sphere);
    const Matrix3Xd noise = generateWhiteNoise(size, stdev);

    pointsOnSphere += noise;

    return pointsOnSphere;
  }
}

// Checks the algebraic approach using six exact points.
TEST(SphereFittingTest, ExactAlgebraic)
{
  Matrix3Xd points(3, 6);

  points << 1.0 , 11.0 ,  6.0 , 6.0 ,  6.0 , 6.0,
            0.0 ,  0.0 , -5.0 , 5.0 ,  0.0 , 0.0,
            0.0 ,  0.0 ,  0.0 , 0.0 , -5.0 , 5.0;

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphere = fitter.run(points);

  EXPECT_NEAR(sphere.radius, 5.0, ProximityTolerance);

  EXPECT_NEAR(sphere.center(0), 6.0, ProximityTolerance);
  EXPECT_NEAR(sphere.center(1), 0.0, ProximityTolerance);
  EXPECT_NEAR(sphere.center(2), 0.0, ProximityTolerance);
}

// Checks the algebraic approach using 10 random but exact points.
TEST(SphereFittingTest, AlgebraicWithTenRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 10;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the algebraic approach using 100 random but exact points.
TEST(SphereFittingTest, AlgebraicWithOneHundredRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the algebraic approach using 1000 random but exact points.
TEST(SphereFittingTest, AlgebraicWithOneThousandRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 1000;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the algebraic  approach using 100 random, noisy points. Noise is
// created from normal distribution with zero mean and standard deviation
// 0.001.
TEST(SphereFittingTest, AlgebraicWithOneHundredRandomNoisyPoints1)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.001);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.0 * 0.001;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// Checks the algebraic  approach using 100 random, noisy points. Noise is
// created from normal distribution with zero mean and standard deviation
// 0.01.
TEST(SphereFittingTest, AlgebraicWithOneHundredRandomNoisyPoints2)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.01);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.0 * 0.01;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// Checks the algebraic  approach using 100 random, noisy points. Noise is
// created from normal distribution with zero mean and standard deviation
// 0.1.
TEST(SphereFittingTest, AlgebraicWithOneHundredRandomNoisyPoints3)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.1);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  constexpr static auto ProximityToleranceApprox = 2.0 * 0.1;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// Checks the linear geometric approach using six exact points.
TEST(SphereFittingTest, ExactLinearGeometric)
{
  Matrix3Xd points(3, 6);

  points << 1.0 , 11.0 ,  6.0 , 6.0 ,  6.0 , 6.0,
            0.0 ,  0.0 , -5.0 , 5.0 ,  0.0 , 0.0,
            0.0 ,  0.0 ,  0.0 , 0.0 , -5.0 , 5.0;

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphere = fitter.run(points);

  EXPECT_NEAR(sphere.radius, 5.0, ProximityTolerance);

  EXPECT_NEAR(sphere.center(0), 6.0, ProximityTolerance);
  EXPECT_NEAR(sphere.center(1), 0.0, ProximityTolerance);
  EXPECT_NEAR(sphere.center(2), 0.0, ProximityTolerance);
}

// Checks the linear geometric approach using 100 random but exact points.
TEST(SphereFittingTest, LinearGeometricWithTenRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 10;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the linear geometric approach using 100 random but exact points.
TEST(SphereFittingTest, LinearGeometricWithOneHundredRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the linear geometric approach using 1000 random but exact points.
TEST(SphereFittingTest, LinearGeometricWithOneThousandRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 1000;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the linear geometric approach using 100 random, noisy points. Noise is
// created from normal distribution  with zero  mean  and standard  deviation
// 0.001.
TEST(SphereFittingTest, LinearGeometricWithOneHundredRandomNoisyPoints1)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.001);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.0 * 0.001;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// Checks the linear geometric approach using 100 random, noisy points. Noise is
// created from normal distribution  with zero  mean  and standard deviation 0.01.
TEST(SphereFittingTest, LinearGeometricWithOneHundredRandomNoisyPoints2)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.01);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.0 * 0.01;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// Checks the linear geometric approach using 100 random, noisy points. Noise is
// created from normal distribution with zero mean and standard deviation 0.1.
TEST(SphereFittingTest, LinearGeometricWithOneHundredRandomNoisyPoints3)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.1);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  constexpr static auto ProximityToleranceApprox = 2.0 * 0.1;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// Checks the nonlinear geometric approach using six exact points.
TEST(SphereFittingTest, ExactNonLinearGeometric)
{
  Matrix3Xd points(3, 6);

  points << 1.0 , 11.0 ,  6.0 , 6.0 ,  6.0 , 6.0,
            0.0 ,  0.0 , -5.0 , 5.0 ,  0.0 , 0.0,
            0.0 ,  0.0 ,  0.0 , 0.0 , -5.0 , 5.0;

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphere = fitter.run(points);

  EXPECT_NEAR(sphere.radius, 5.0, ProximityTolerance);

  EXPECT_NEAR(sphere.center(0), 6.0, ProximityTolerance);
  EXPECT_NEAR(sphere.center(1), 0.0, ProximityTolerance);
  EXPECT_NEAR(sphere.center(2), 0.0, ProximityTolerance);
}

// Checks the nonlinear geometric approach using 10 random but exact points.
TEST(SphereFittingTest, NonLinearWithTenRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 10;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the nonlinear approach using 100 random but exact points.
TEST(SphereFittingTest, NonLinearGeometricWithOneHundredRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the nonlinear approach using 1000 random but exact points.
TEST(SphereFittingTest, NonLinearGeometricWithOneThousandRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 1000;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// Checks the nonlinear approach using 100 random, noisy points. Noise is
// created from normal distribution with zero mean and standard deviation
// 0.001.
TEST(SphereFittingTest, NonLinearGeometricWithOneHundredRandomNoisyPoints1)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.001);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.0 * 0.001;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// Checks the nonlinear approach using 100 random, noisy points. Noise is
// created from normal distribution with zero mean and standard deviation
// 0.01.
TEST(SphereFittingTest, NonLinearGeometricWithOneHundredRandomNoisyPoints2)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.01);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.0 * 0.01;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// Checks the nonlinear approach using 100 random, noisy points. Noise is
// created from normal distribution with zero mean and standard deviation
// 0.1.
TEST(SphereFittingTest, NonLinearGeometricWithOneHundredRandomNoisyPoints3)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.0, -2.0, 4.0;
  sphereExact.radius = 8.0;
    
  Matrix3Xd points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.1);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  constexpr static auto ProximityToleranceApprox = 2.0 * 0.1;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}
