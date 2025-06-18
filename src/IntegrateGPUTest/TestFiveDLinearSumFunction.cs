//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestFiveDLinearSumFunction
    {
        [TestMethod]
        public void TestX1PlusX2PlusX3PlusX4PlusX5Function()
        {
            // f(x1,x2,x3,x4,x5) = x1+x2+x3+x4+x5, integral over [0,1]^5 is 2.5
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                torch.Tensor x1_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor x2_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                torch.Tensor x3_coords = samples.index(torch.TensorIndex.Ellipsis, 2);
                torch.Tensor x4_coords = samples.index(torch.TensorIndex.Ellipsis, 3);
                torch.Tensor x5_coords = samples.index(torch.TensorIndex.Ellipsis, 4);
                return x1_coords.add(x2_coords).add(x3_coords).add(x4_coords).add(x5_coords);
            };

            torch.Tensor lowerBounds = torch.tensor(new double[] { 0.0D, 0.0D, 0.0D, 0.0D, 0.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            torch.Tensor upperBounds = torch.tensor(new double[] { 1.0D, 1.0D, 1.0D, 1.0D, 1.0D }, dtype: ScalarType.Float64, device: torch.CPU);
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

            double expectedIntegral = 2.5D; // 5 * (0.5)
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x1..x5)=x1+..+x5 did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}