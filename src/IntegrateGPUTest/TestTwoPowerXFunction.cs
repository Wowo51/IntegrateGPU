//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestTwoPowerXFunction
    {
        [TestMethod]
        public void TestTwoPowerX()
        {
            // f(x) = 2^x, integral from 0 to 1 is 1/ln(2)
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.pow(2.0D, x);
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

            double expectedIntegral = 1.0D / Math.Log(2.0D);
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=2^x did not match expected value.");
        }
    }
}