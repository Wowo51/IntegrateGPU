//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestThreeDLinearFunction
    {
        [TestMethod]
        public void TestXPlusYPlusZFunction()
        {
            // f(x, y, z) = x + y + z, integral over x=[0,1], y=[0,1], z=[0,1] is 3/2
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                // samples shape: [numberOfSamples, 3]
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                torch.Tensor z_coords = samples.index(torch.TensorIndex.Ellipsis, 2);
                return x_coords.add(y_coords).add(z_coords);
            };

            torch.Tensor lowerBounds = torch.tensor(new double[] { 0.0D, 0.0D, 0.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            torch.Tensor upperBounds = torch.tensor(new double[] { 1.0D, 1.0D, 1.0D }, dtype: ScalarType.Float64, device: torch.CPU);
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

            double expectedIntegral = 1.5D; // 3/2
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z)=x+y+z did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}