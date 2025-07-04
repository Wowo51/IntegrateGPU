//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestArctanBaseFunction
    {
        [TestMethod]
        public void TestOneOverOnePlusXSquared()
        {
            // f(x) = 1/(1+x^2), integral from 0 to 1 is PI/4 (arctan(1) - arctan(0))
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.reciprocal(1.0D + torch.pow(x, 2.0D));
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

            double expectedIntegral = Math.PI / 4.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=1/(1+x^2) did not match expected value.");
        }
    }
}