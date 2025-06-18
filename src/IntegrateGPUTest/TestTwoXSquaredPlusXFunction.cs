//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestTwoXSquaredPlusXFunction
    {
        [TestMethod]
        public void TestTwoXSquaredPlusX()
        {
            // f(x) = 2x^2 + x, integral from 0 to 1 is 7/6
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.pow(x, 2.0D).mul(2.0D).add(x);
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

            double expectedIntegral = 7.0D / 6.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=2x^2+x did not match expected value.");
        }
    }
}