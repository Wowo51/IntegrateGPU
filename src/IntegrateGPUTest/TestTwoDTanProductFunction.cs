//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestTwoDTanProductFunction
    {
        [TestMethod]
        public void TestTanXTimesTanYFunction()
        {
            // f(x, y) = tan(x) * tan(y), integral over x=[0,PI/4], y=[0,PI/4] is (ln(sqrt(2)))^2
            // Integral of tan(u) is -ln(cos(u))
            // From 0 to PI/4: -ln(cos(PI/4)) - (-ln(cos(0))) = -ln(1/sqrt(2)) - (-ln(1)) = ln(sqrt(2)) - 0 = 0.5 * ln(2)
            // Product integral: (0.5 * ln(2))^2
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                return torch.tan(x_coords).mul(torch.tan(y_coords));
            };

            torch.Tensor lowerBounds = torch.tensor(new double[] { 0.0D, 0.0D }, dtype: ScalarType.Float64, device: torch.CPU);
            torch.Tensor upperBounds = torch.tensor(new double[] { Math.PI / 4.0D, Math.PI / 4.0D }, dtype: ScalarType.Float64, device: torch.CPU);
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

            double expectedIntegral = Math.Pow(0.5D * Math.Log(2.0D), 2.0D);
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y)=tan(x)*tan(y) did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}