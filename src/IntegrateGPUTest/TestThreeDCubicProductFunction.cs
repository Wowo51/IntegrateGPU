//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestThreeDCubicProductFunction
    {
        [TestMethod]
        public void TestXCubedTimesYCubedTimesZCubedFunction()
        {
            // f(x, y, z) = x^3 * y^3 * z^3, integral over [0,1]^3 is (1/4)^3 = 1/64 = 0.015625
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                torch.Tensor z_coords = samples.index(torch.TensorIndex.Ellipsis, 2);
                return torch.pow(x_coords, 3.0D).mul(torch.pow(y_coords, 3.0D)).mul(torch.pow(z_coords, 3.0D));
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

            double expectedIntegral = Math.Pow(0.25D, 3.0D); // (Integral of x^3 from 0 to 1)^3 = (1/4)^3
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z)=x^3*y^3*z^3 did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}