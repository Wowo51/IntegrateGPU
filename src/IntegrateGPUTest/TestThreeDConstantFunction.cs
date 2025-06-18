//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestThreeDConstantFunction
    {
        [TestMethod]
        public void TestTenInThreeD()
        {
            // f(x, y, z) = 10, integral over x=[0,1], y=[0,1], z=[0,1] is 10
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                // samples shape: [numberOfSamples, 3]
                // Return a tensor of the same sample size, filled with the constant value 10.0D.
                // The input 'samples' tensor is used to determine the output shape's first dimension.
                return torch.full(new long[] { samples.shape[0] }, 10.0D, dtype: ScalarType.Float64);
            };

            torch.Tensor lowerBounds = torch.tensor(new double[] { 0.0D, 0.0D, 0.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            torch.Tensor upperBounds = torch.tensor(new double[] { 1.0D, 1.0D, 1.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            long numberOfSamples = 2_000_000L;
            torch.Device device = torch.CPU;

            double estimatedIntegral = 0.0D;
            try
            {
                estimatedIntegral = IntegrateGPU.IntegrateGPU.IntegrateMonteCarlo(
                    func,
                    lowerBounds,
                    upperBounds,
                    numberOfSamples,
                    device);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Test failed with an exception: {ex.Message}{Environment.NewLine}{ex.StackTrace}");
            }

            double expectedIntegral = 10.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z)=10 did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}