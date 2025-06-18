//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestLinearFunction
    {
        [TestMethod]
        public void TestXFunction()
        {
            // f(x) = x, integral from 0 to 1 is 0.5
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return x;
            };

            double lowerBound = 0.0D;
            double upperBound = 1.0D;
            long numberOfSamples = 1_000_000L; // Use a large number of samples for better accuracy
            torch.Device device = torch.CPU; // Use CPU for testing

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

            double expectedIntegral = 0.5D;
            double tolerance = 0.01D; // Monte Carlo is probabilistic, allow for tolerance

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=x did not match expected value.");
        }
    }
}