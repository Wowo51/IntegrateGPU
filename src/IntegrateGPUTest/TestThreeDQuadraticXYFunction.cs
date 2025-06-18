//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestThreeDQuadraticXYFunction
    {
        [TestMethod]
        public void TestXTimesYSquaredPlusZFunction()
        {
            // f(x,y,z) = x*y^2 + z, integral over x=[0,1], y=[0,1], z=[0,1] is 2/3
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                torch.Tensor z_coords = samples.index(torch.TensorIndex.Ellipsis, 2);
                return x_coords.mul(torch.pow(y_coords, 2.0D)).add(z_coords);
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

            double expectedIntegral = (1.0D/2.0D * 1.0D/3.0D) + (1.0D/2.0D); // Integral of x*y^2 is (x^2/2)*(y^3/3) = 1/6. Integral of z is z^2/2 = 1/2. Total = 1/6 + 1/2 = 4/6 = 2/3
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z)=x*y^2+z did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}