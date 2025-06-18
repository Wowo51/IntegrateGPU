//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestThreeDLinearCombinationFunction
    {
        [TestMethod]
        public void TestTwoXPlusThreeYPlusFourZFunction()
        {
            // f(x,y,z) = 2x + 3y + 4z, integral over [0,1]^3 is 4.5
            // Integral of 2x from 0 to 1 is 2 * (x^2/2) | 0-1 = 1
            // Integral of 3y from 0 to 1 is 3 * (y^2/2) | 0-1 = 1.5
            // Integral of 4z from 0 to 1 is 4 * (z^2/2) | 0-1 = 2
            // Total integral = 1 + 1.5 + 2 = 4.5
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                torch.Tensor z_coords = samples.index(torch.TensorIndex.Ellipsis, 2);
                return x_coords.mul(2.0D).add(y_coords.mul(3.0D)).add(z_coords.mul(4.0D));
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

            double expectedIntegral = 4.5D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z)=2x+3y+4z did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}