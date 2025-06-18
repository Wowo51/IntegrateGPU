//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestReciprocalXSquaredFunction
    {
        [TestMethod]
        public void TestReciprocalXSquared()
        {
            // f(x) = 1/x^2, integral from 1 to 2 is 0.5
            // Integral of 1/x^2 is -1/x
            // Evaluate from 1 to 2: (-1/2) - (-1/1) = -0.5 + 1 = 0.5
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.pow(x, -2.0D);
            };

            double lowerBound = 1.0D;
            double upperBound = 2.0D;
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

            double expectedIntegral = 0.5D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=1/x^2 did not match expected value.");
        }
    }
}