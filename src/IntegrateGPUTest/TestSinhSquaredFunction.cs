//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestSinhSquaredFunction
    {
        [TestMethod]
        public void TestSinhSquaredX()
        {
            // f(x) = sinh^2(x), integral from 0 to 1 is 1/2 - sinh(2)/4
            // Integral of sinh^2(x) dx = x/2 - sinh(2x)/4
            // From 0 to 1: (1/2 - sinh(2)/4) - (0 - sinh(0)/4) = 1/2 - sinh(2)/4
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.sinh(x).pow(2.0D);
            };

            double lowerBound = 0.0D;
            double upperBound = 1.0D;
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

            double expectedIntegral = Math.Sinh(2.0D) / 4.0D - 0.5D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=sinh^2(x) did not match expected value.");
        }
    }
}