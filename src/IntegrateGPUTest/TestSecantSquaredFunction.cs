//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestSecantSquaredFunction
    {
        [TestMethod]
        public void TestSecantSquaredX()
        {
            // f(x) = sec^2(x) = 1 / cos^2(x), integral from 0 to PI/4 is 1
            // Integral of sec^2(x) is tan(x)
            // From 0 to PI/4: tan(PI/4) - tan(0) = 1 - 0 = 1
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.pow(torch.cos(x), -2.0D); // sec^2(x)
            };

            double lowerBound = 0.0D;
            double upperBound = Math.PI / 4.0D;
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

            double expectedIntegral = 1.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=sec^2(x) did not match expected value.");
        }
    }
}