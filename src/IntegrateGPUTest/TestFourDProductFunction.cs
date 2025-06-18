//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestFourDProductFunction
    {
        [TestMethod]
        public void TestXTimesYTimesZTimesWFunction()
        {
            // f(x,y,z,w) = x*y*z*w, integral over [0,1]^4 is (1/2)^4 = 1/16 = 0.0625
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                torch.Tensor z_coords = samples.index(torch.TensorIndex.Ellipsis, 2);
                torch.Tensor w_coords = samples.index(torch.TensorIndex.Ellipsis, 3);
                return x_coords.mul(y_coords).mul(z_coords).mul(w_coords);
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

            double expectedIntegral = 0.0625D;
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z,w)=x*y*z*w did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}