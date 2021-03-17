#include "SphereFitting.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <vector>

#include <unsupported/Eigen/NonLinearOptimization>

using namespace MatchingTools;

namespace
{
  using Matrix3Xf = SphereFitting::Matrix3Xf;
  using VectorXf = Eigen::VectorXf;
  using CenterRadiusPair = SphereFitting::CenterRadiusPair;

  constexpr static auto ProximityTolerance = 0.00001f;

  std::default_random_engine generator;

  Matrix3Xf generateWhiteNoise(int i_size, float i_stdev)
  {
    Matrix3Xf noise(3, i_size);

    std::normal_distribution<float> distrX(0.f, i_stdev);
    std::normal_distribution<float> distrY(0.f, i_stdev);
    std::normal_distribution<float> distrZ(0.f, i_stdev);
    
    for (Eigen::Index col = 0; col < noise.cols(); ++col)
      noise.col(col) << distrX(generator), distrY(generator), distrZ(generator);

    return noise;
  }

  // This simple method is described at
  //
  // Mervin  E.  Muller,  A Note  on  a Method  for Generating  Points
  // Uniformly  on N-Dimensional  Spheres Communications  of the  ACM,
  // 2(4), p. 19-20, 1959
  
  Matrix3Xf generateUniformlyDistributedPointsOnSphere(int i_size, const CenterRadiusPair& i_sphere)
  {
    Matrix3Xf samples(3, i_size);
    
    std::normal_distribution<float> distrX(0.f, 1.f);
    std::normal_distribution<float> distrY(0.f, 1.f);
    std::normal_distribution<float> distrZ(0.f, 1.f);
    
    for (Eigen::Index col = 0; col < samples.cols(); ++col)
    {
      samples.col(col) << distrX(generator), distrY(generator), distrZ(generator);
      samples.col(col) = i_sphere.radius * samples.col(col).normalized() + i_sphere.center;      
    }

    return samples;
  }

  Matrix3Xf generatePointsCloseToSphere(int i_size, const CenterRadiusPair& i_sphere, float i_stdev)
  {
    Matrix3Xf pointsOnSphere = generateUniformlyDistributedPointsOnSphere(i_size, i_sphere);
    const Matrix3Xf noise = generateWhiteNoise(i_size, i_stdev);

    pointsOnSphere += noise;

    return pointsOnSphere;
  }
}

// [TDR-1668] - Checks the algebraic approach using six exact points.
TEST(SphereFittingTest, ExactAlgebraic)
{
  Matrix3Xf points(3, 6);

  points << 1.f , 11.f ,  6.f , 6.f ,  6.f , 6.f,
            0.f ,  0.f , -5.f , 5.f ,  0.f , 0.f,
            0.f ,  0.f ,  0.f , 0.f , -5.f , 5.f;

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphere = fitter.run(points);

  EXPECT_NEAR(sphere.radius, 5.f, ProximityTolerance);

  EXPECT_NEAR(sphere.center(0), 6.f, ProximityTolerance);
  EXPECT_NEAR(sphere.center(1), 0.f, ProximityTolerance);
  EXPECT_NEAR(sphere.center(2), 0.f, ProximityTolerance);
}

// [TDR-1668] - Checks the algebraic approach using 10 random but exact points.
TEST(SphereFittingTest, AlgebraicWithTenRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 10;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the algebraic approach using 100 random but exact points.
TEST(SphereFittingTest, AlgebraicWithOneHundredRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the algebraic approach using 1000 random but exact points.
TEST(SphereFittingTest, AlgebraicWithOneThousandRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 1000;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the algebraic  approach using 100 random, noisy points. Noise
// is created from normal distribution with zero mean and standard deviation 0.001.
TEST(SphereFittingTest, AlgebraicWithOneHundredRandomNoisyPoints1)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.001f);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.f * 0.001f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// [TDR-1668] - Checks the algebraic  approach using 100 random, noisy points. Noise
// is created from normal distribution with zero mean and standard deviation 0.01.
TEST(SphereFittingTest, AlgebraicWithOneHundredRandomNoisyPoints2)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.01f);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.f * 0.01f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// [TDR-1668] - Checks the algebraic  approach using 100 random, noisy points. Noise
// is created from normal distribution with zero mean and standard deviation 0.1.
TEST(SphereFittingTest, AlgebraicWithOneHundredRandomNoisyPoints3)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.1f);

  SphereFitting fitter(SphereFittingAlgorithm::Algebraic);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  constexpr static auto ProximityToleranceApprox = 2.f * 0.1f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// [TDR-1668] - Checks the linear geometric approach using six exact points.
TEST(SphereFittingTest, ExactLinearGeometric)
{
  Matrix3Xf points(3, 6);

  points << 1.f , 11.f ,  6.f , 6.f ,  6.f , 6.f,
            0.f ,  0.f , -5.f , 5.f ,  0.f , 0.f,
            0.f ,  0.f ,  0.f , 0.f , -5.f , 5.f;

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphere = fitter.run(points);

  EXPECT_NEAR(sphere.radius, 5.f, ProximityTolerance);

  EXPECT_NEAR(sphere.center(0), 6.f, ProximityTolerance);
  EXPECT_NEAR(sphere.center(1), 0.f, ProximityTolerance);
  EXPECT_NEAR(sphere.center(2), 0.f, ProximityTolerance);
}

// [TDR-1668] - Checks the linear geometric approach using 100 random but exact points.
TEST(SphereFittingTest, LinearGeometricWithTenRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 10;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the linear geometric approach using 100 random but exact points.
TEST(SphereFittingTest, LinearGeometricWithOneHundredRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the linear geometric approach using 1000 random but exact points.
TEST(SphereFittingTest, LinearGeometricWithOneThousandRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 1000;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the linear geometric approach using 100 random, noisy points.
// Noise is created from normal distribution  with zero  mean  and standard  deviation
// 0.001.
TEST(SphereFittingTest, LinearGeometricWithOneHundredRandomNoisyPoints1)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.001f);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.f * 0.001f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// [TDR-1668] - Checks the linear geometric approach using 100 random, noisy points.
// Noise is created from normal distribution  with zero  mean  and standard deviation
// 0.01.
TEST(SphereFittingTest, LinearGeometricWithOneHundredRandomNoisyPoints2)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.01f);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.f * 0.01f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// [TDR-1668] - Checks the linear geometric approach using 100 random, noisy points.
// Noise is created from normal distribution with zero mean and standard deviation
// 0.1.
TEST(SphereFittingTest, LinearGeometricWithOneHundredRandomNoisyPoints3)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.1f);

  SphereFitting fitter(SphereFittingAlgorithm::LinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  constexpr static auto ProximityToleranceApprox = 2.f * 0.1f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// [TDR-1668] - Checks the nonlinear geometric approach using six exact points.
TEST(SphereFittingTest, ExactNonLinearGeometric)
{
  Matrix3Xf points(3, 6);

  points << 1.f , 11.f ,  6.f , 6.f ,  6.f , 6.f,
            0.f ,  0.f , -5.f , 5.f ,  0.f , 0.f,
            0.f ,  0.f ,  0.f , 0.f , -5.f , 5.f;

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphere = fitter.run(points);

  EXPECT_NEAR(sphere.radius, 5.f, ProximityTolerance);

  EXPECT_NEAR(sphere.center(0), 6.f, ProximityTolerance);
  EXPECT_NEAR(sphere.center(1), 0.f, ProximityTolerance);
  EXPECT_NEAR(sphere.center(2), 0.f, ProximityTolerance);
}

// [TDR-1668] - Checks the nonlinear geometric approach using 10 random but exact points.
TEST(SphereFittingTest, NonLinearWithTenRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 10;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the nonlinear approach using 100 random but exact points.
TEST(SphereFittingTest, NonLinearGeometricWithOneHundredRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the nonlinear approach using 1000 random but exact points.
TEST(SphereFittingTest, NonLinearGeometricWithOneThousandRandomExactPoints)
{
  constexpr static auto NumberOfPoints = 1000;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generateUniformlyDistributedPointsOnSphere(NumberOfPoints, sphereExact);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityTolerance);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityTolerance);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityTolerance);
}

// [TDR-1668] - Checks the nonlinear approach using 100 random, noisy points. Noise
// is created from normal distribution with zero mean and standard deviation 0.001.
TEST(SphereFittingTest, NonLinearGeometricWithOneHundredRandomNoisyPoints1)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.001f);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.f * 0.001f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// [TDR-1668] - Checks the nonlinear approach using 100 random, noisy points. Noise
// is created from normal distribution with zero mean and standard deviation 0.01.
TEST(SphereFittingTest, NonLinearGeometricWithOneHundredRandomNoisyPoints2)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.01f);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

 constexpr static auto ProximityToleranceApprox = 2.f * 0.01f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}

// [TDR-1668] - Checks the nonlinear approach using 100 random, noisy points. Noise
// is created from normal distribution with zero mean and standard deviation 0.1.
TEST(SphereFittingTest, NonLinearGeometricWithOneHundredRandomNoisyPoints3)
{
  constexpr static auto NumberOfPoints = 100;
  
  CenterRadiusPair sphereExact;
  sphereExact.center << 1.f, -2.f, 4.f;
  sphereExact.radius = 8.f;
    
  Matrix3Xf points = generatePointsCloseToSphere(NumberOfPoints, sphereExact, 0.1f);

  SphereFitting fitter(SphereFittingAlgorithm::NonLinearGeometric);
  fitter.run(points);

  const auto sphereApprox = fitter.run(points);

  constexpr static auto ProximityToleranceApprox = 2.f * 0.1f;

  EXPECT_NEAR(sphereApprox.radius, sphereExact.radius, ProximityToleranceApprox);

  EXPECT_NEAR(sphereApprox.center(0), sphereExact.center(0), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(1), sphereExact.center(1), ProximityToleranceApprox);
  EXPECT_NEAR(sphereApprox.center(2), sphereExact.center(2), ProximityToleranceApprox);
}
