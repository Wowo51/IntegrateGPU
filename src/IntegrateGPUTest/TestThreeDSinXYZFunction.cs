//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestThreeDSinXYZFunction
    {
        [TestMethod]
        public void TestSinXSinYSinZFunction()
        {
            // f(x, y, z) = sin(x)sin(y)sin(z), integral over x=[0,PI/2], y=[0,PI/2], z=[0,PI/2] is 1
            // Integral of sin(u) is -cos(u).
            // For one dimension: Integral from 0 to PI/2 of sin(x) dx is [-cos(x)] from 0 to PI/2 = (-cos(PI/2)) - (-cos(0)) = 0 - (-1) = 1.
            // Since it's separable: (Integral sin(x) dx) * (Integral sin(y) dy) * (Integral sin(z) dz) = 1 * 1 * 1 = 1.
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                torch.Tensor z_coords = samples.index(torch.TensorIndex.Ellipsis, 2);
                return torch.sin(x_coords).mul(torch.sin(y_coords)).mul(torch.sin(z_coords));
            };

            torch.Tensor lowerBounds = torch.tensor(new double[] { 0.0D, 0.0D, 0.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            torch.Tensor upperBounds = torch.tensor(new double[] { Math.PI / 2.0D, Math.PI / 2.0D, Math.PI / 2.0D }, dtype: ScalarType.Float64, device: torch.CPU);
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

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z)=sin(x)sin(y)sin(z) did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}