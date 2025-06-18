//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestSineFunction
    {
        [TestMethod]
        public void TestSineXFunction()
        {
            // f(x) = sin(x), integral from 0 to PI is 2
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor x) =>
            {
                return torch.sin(x);
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

            double expectedIntegral = 2.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x)=sin(x) did not match expected value.");
        }
    }
}