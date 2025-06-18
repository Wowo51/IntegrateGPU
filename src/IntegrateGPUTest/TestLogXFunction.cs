//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestLogXFunction
    {
        [TestMethod]
        public void TestLogX()
        {
            // f(x) = ln(x), integral from 1 to e^2 is e^2 + 1
            // Integral of ln(x) dx is xln(x) - x
            // Evaluate from 1 to e^2: (e^2 * ln(e^2) - e^2) - (1 * ln(1) - 1)
            // = (e^2 * 2 - e^2) - (0 - 1)
            // = (2e^2 - e^2) + 1
            // = e^2 + 1
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.log(x);
            };

            double lowerBound = 1.0D;
            double upperBound = Math.Pow(Math.E, 2.0D); // e^2
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

            double expectedIntegral = Math.Pow(Math.E, 2.0D) + 1.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=ln(x) did not match expected value.");
        }
    }
}