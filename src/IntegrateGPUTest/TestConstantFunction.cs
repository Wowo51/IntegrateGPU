//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestConstantFunction
    {
        [TestMethod]
        public void TestFive()
        {
            // f(x) = 5, integral from 0 to 3 is 15
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                // For a constant function, return a tensor of the same leading dimension size as input, filled with the constant value.
                return torch.full(new long[] { x.shape[0] }, 5.0D, dtype: ScalarType.Float64);
            };

            double lowerBound = 0.0D;
            double upperBound = 3.0D;
            long numberOfSamples = 1_000_000L;
            torch.Device device = torch.CPU;

            double estimatedIntegral = 0.0D;
            try
            {
                estimatedIntegral = IntegrateGPU.IntegrateGPU.IntegrateMonteCarlo(
                    func,
                    lowerBound,
                    upperBound,
                    numberOfSamples,
                    device);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Test failed with an exception: {ex.Message}{Environment.NewLine}{ex.StackTrace}");
            }

            double expectedIntegral = 15.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=5 did not match expected value.");
        }
    }
}