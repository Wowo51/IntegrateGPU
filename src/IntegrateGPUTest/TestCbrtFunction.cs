//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestCbrtFunction
    {
        [TestMethod]
        public void TestCbrtXFunction()
        {
            // f(x) = x^(1/3), integral from 0 to 8 is 12
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.pow(x, 1.0D / 3.0D);
            };

            double lowerBound = 0.0D;
            double upperBound = 8.0D;
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

            double expectedIntegral = 12.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=x^(1/3) did not match expected value.");
        }
    }
}