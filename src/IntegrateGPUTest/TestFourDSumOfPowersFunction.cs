//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using IntegrateGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace IntegrateGPUTest
{
    [TestClass]
    public sealed class TestFourDSumOfPowersFunction
    {
        [TestMethod]
        public void TestXPlusYSquaredPlusZCubedPlusWToTheFourthFunction()
        {
            // f(x,y,z,w) = x + y^2 + z^3 + w^4, integral over [0,1]^4 is 1/2 + 1/3 + 1/4 + 1/5 = 77/60
            Func<torch.Tensor, torch.Tensor> func = (torch.Tensor samples) =>
            {
                torch.Tensor x_coords = samples.index(torch.TensorIndex.Ellipsis, 0);
                torch.Tensor y_coords = samples.index(torch.TensorIndex.Ellipsis, 1);
                torch.Tensor z_coords = samples.index(torch.TensorIndex.Ellipsis, 2);
                torch.Tensor w_coords = samples.index(torch.TensorIndex.Ellipsis, 3);
                return x_coords.add(torch.pow(y_coords, 2.0D)).add(torch.pow(z_coords, 3.0D)).add(torch.pow(w_coords, 4.0D));
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

            double expectedIntegral = (1.0D/2.0D) + (1.0D/3.0D) + (1.0D/4.0D) + (1.0D/5.0D); // Sum of integrals for each term
            double tolerance = 0.01D;

            Assert.AreEqual(expectedIntegral, estimatedIntegral, tolerance, "Integral of f(x,y,z,w)=x+y^2+z^3+w^4 did not match expected value.");

            lowerBounds.Dispose();
            upperBounds.Dispose();
        }
    }
}