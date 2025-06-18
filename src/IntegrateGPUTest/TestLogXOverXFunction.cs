//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestLogXOverXFunction
    {
        [TestMethod]
        public void TestLogXOverX()
        {
            // f(x) = ln(x) / x, integral from 1 to e is 0.5
            // Let u = ln(x), du = 1/x dx. Integral is u du = u^2 / 2 = (ln(x))^2 / 2
            // From 1 to e: (ln(e))^2 / 2 - (ln(1))^2 / 2 = 1^2 / 2 - 0 = 0.5
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.log(x).div(x);
            };

            double lowerBound = 1.0D;
            double upperBound = Math.E;
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

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=ln(x)/x did not match expected value.");
        }
    }
}