//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestFourDConstantFunction
    {
        [TestMethod]
        public void TestSevenInFourD()
        {
            // f(x, y, z, w) = 7, integral over x=[0,1], y=[0,1], z=[0,1], w=[0,1] is 7
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                // samples shape: [numberOfSamples, 4]
                return torch.full(new long[] { samples.shape[0] }, 7.0D, dtype: ScalarType.Float64);
            };

            torch.Tensor lowerBounds = torch.tensor(new double[] { 0.0D, 0.0D, 0.0D, 0.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            torch.Tensor upperBounds = torch.tensor(new double[] { 1.0D, 1.0D, 1.0D, 1.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            long numberOfSamples = 2_000_000L;
            torch.Device device = torch.CPU;

            double estimatedIntegral = 0.0D;
            try
            {
                estimatedIntegral = IntegrateGPU.IntegrateGPU.IntegrateMonteCarlo(
                    func,
                    lowerBounds,
                    upperBounds,
                    numberOfSamples,
                    device);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Test failed with an exception: {ex.Message}{Environment.NewLine}{ex.StackTrace}");
            }


            double expectedIntegral = 7.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z,w)=7 did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}