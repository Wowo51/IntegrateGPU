//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestSinSquaredXFunction
    {
        [TestMethod]
        public void TestSinSquaredX()
        {
            // f(x) = sin^2(x), integral from 0 to PI is PI/2
            // Integral of sin^2(x) = x/2 - sin(2x)/4
            // Evaluate from 0 to PI: (PI/2 - sin(2*PI)/4) - (0/2 - sin(0)/4)
            // = (PI/2 - 0) - (0 - 0) = PI/2
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.sin(x).pow(2.0D);
            };

            double lowerBound = 0.0D;
            double upperBound = Math.PI;
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

            double expectedIntegral = Math.PI / 2.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=sin^2(x) did not match expected value.");
        }
    }
}