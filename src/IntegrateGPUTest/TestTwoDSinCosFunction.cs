//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestTwoDSinCosFunction
    {
        [TestMethod]
        public void TestSinXCosYFunction()
        {
            // f(x, y) = sin(x)cos(y), integral over x=[0,PI/2], y=[0,PI/2] is 1
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                // samples shape: [numberOfSamples, 2]
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                return torch.sin(x_coords).mul(torch.cos(y_coords));
            };

            torch.Tensor lowerBounds = torch.tensor(new double[] { 0.0D, 0.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            torch.Tensor upperBounds = torch.tensor(new double[] { Math.PI / 2.0D, Math.PI / 2.0D }, dtype: ScalarType.Float64, device: torch.CPU);
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

            double expectedIntegral = 1.0D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y)=sin(x)cos(y) did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}