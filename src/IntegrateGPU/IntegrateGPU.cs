//Copyright Warren Harding 2025.
using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace IntegrateGPU
{
    public static class IntegrateGPU
    {
        /// <summary>
        /// Performs numerical integration using the Monte Carlo method for a 1D function.
        /// This method dispatches to the N-dimensional integration by converting scalar bounds to 1D tensors.
        /// </summary>
        /// <param name="functionToIntegrate">The function to integrate. It should accept a TorchSharp.torch.Tensor of random samples
        /// and return a TorchSharp.torch.Tensor containing the function values for each sample.
        /// The input tensor will be a 1D tensor of double values representing the random x-coordinates.
        /// The output tensor should also be a 1D tensor of double values with the same number of elements as the input.</param>
        /// <param name="lowerBound">The lower bound of the integration interval.</param>
        /// <param name="upperBound">The upper bound of the integration interval.</param>
        /// <param name="numberOfSamples">The number of random samples to use for the Monte Carlo integration. Must be greater than 0.</param>
        /// <param name="device">The TorchSharp device to use for computations (e.g., torch.CPU or torch.CUDA).</param>
        /// <returns>The estimated value of the definite integral, or 0.0 if input parameters are invalid or an error occurs during computation.</returns>
        public static double IntegrateMonteCarlo(
            Func<torch.Tensor, torch.Tensor> functionToIntegrate,
            double lowerBound,
            double upperBound,
            long numberOfSamples,
            torch.Device device)
        {
            if (numberOfSamples <= 0)
            {
                Console.Error.WriteLine("IntegrateMonteCarlo (1D): numberOfSamples must be greater than 0.");
                return 0.0;
            }

            try
            {
                // Convert scalar bounds to 1D tensors
                using (torch.Tensor lowerBoundTensor = torch.tensor(new double[] { lowerBound }, dtype: torch.float64, device: device))
                using (torch.Tensor upperBoundTensor = torch.tensor(new double[] { upperBound }, dtype: torch.float64, device: device))
                {
                    // Call the N-dimensional integration method
                    return IntegrateMonteCarlo(functionToIntegrate, lowerBoundTensor, upperBoundTensor, numberOfSamples, device);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"IntegrateMonteCarlo (1D) caught an exception during tensor creation or dispatch: {ex.Message}{Environment.NewLine}{ex.StackTrace}");
                return 0.0;
            }
        }

        /// <summary>
        /// Performs numerical integration using the Monte Carlo method for an N-dimensional function.
        /// </summary>
        /// <param name="functionToIntegrate">The function to integrate. It should accept a TorchSharp.torch.Tensor of random samples
        /// where each row is a sample point and columns represent dimensions, and return a TorchSharp.torch.Tensor
        /// containing the function values for each sample.</param>
        /// <param name="lowerBounds">A 1D TorchSharp.torch.Tensor containing the lower bounds for each dimension.</param>
        /// <param name="upperBounds">A 1D TorchSharp.torch.Tensor containing the upper bounds for each dimension.</param>
        /// <param name="numberOfSamples">The number of random samples to use for the Monte Carlo integration. Must be greater than 0.</param>
        /// <param name="device">The TorchSharp device to use for computations (e.g., torch.CPU or torch.CUDA).</param>
        /// <returns>The estimated value of the definite integral, or 0.0 if input parameters are invalid or an error occurs during computation.</returns>
        public static double IntegrateMonteCarlo(
            Func<torch.Tensor, torch.Tensor> functionToIntegrate,
            torch.Tensor lowerBounds,
            torch.Tensor upperBounds,
            long numberOfSamples,
            torch.Device device)
        {
            if (numberOfSamples <= 0)
            {
                Console.Error.WriteLine("IntegrateMonteCarlo (ND): numberOfSamples must be greater than 0.");
                return 0.0;
            }

            if (lowerBounds.Dimensions != 1 || upperBounds.Dimensions != 1 || lowerBounds.shape[0] != upperBounds.shape[0])
            {
                Console.Error.WriteLine("IntegrateMonteCarlo (ND): lowerBounds and upperBounds must be 1D tensors of the same size.");
                return 0.0; // Bounds must be 1D tensors of the same size.
            }

            long dimensionality = lowerBounds.shape[0];
            if (dimensionality == 0) // Handle zero dimensions gracefully
            {
                Console.Error.WriteLine("IntegrateMonteCarlo (ND): Dimensionality is zero.");
                return 0.0;
            }

            try
            {
                using (torch.Tensor lowerBoundTensor = lowerBounds.to(device))
                using (torch.Tensor upperBoundTensor = upperBounds.to(device))
                {
                    using (torch.Tensor ranges = upperBoundTensor - lowerBoundTensor)
                    {
                        // Check if any upper bound is less than its corresponding lower bound
                        if (torch.any(ranges < 0.0).item<bool>())
                        {
                            Console.Error.WriteLine("IntegrateMonteCarlo (ND): An upper bound is less than its corresponding lower bound.");
                            return 0.0;
                        }

                        using (torch.Tensor volumeTensor = torch.prod(ranges))
                        {
                            double integrationVolume = volumeTensor.item<double>();

                            if (integrationVolume <= 0.0) // Handle zero or negative volume
                            {
                                Console.Error.WriteLine("IntegrateMonteCarlo (ND): Integration volume is zero or negative.");
                                return 0.0;
                            }

                            // Generate random samples within the [0, 1) hypercube for each dimension
                            // Shape: (numberOfSamples, dimensionality)
                            using (torch.Tensor samplesUnitCube = torch.rand(new long[] { numberOfSamples, dimensionality }, dtype: torch.float64, device: device))
                            {
                                // Scale and shift samples to fit within the custom bounds [lower, upper]
                                // scaled_samples = lower + rand * (upper - lower)
                                // TorchSharp automatically handles broadcasting for scalar ranges and lower/upper bounds across samples.
                                using (torch.Tensor scaledSamples = lowerBoundTensor + samplesUnitCube * ranges)
                                {
                                    // Evaluate the function at the sampled points
                                    using (torch.Tensor functionValues = functionToIntegrate(scaledSamples))
                                    {
                                        // Ensure functionValues is not null and has values for mean calculation
                                        if (functionValues is null || functionValues.numel() == 0)
                                        {
                                            Console.Error.WriteLine("IntegrateMonteCarlo (ND): Function returned null or an empty tensor.");
                                            return 0.0;
                                        }

                                        // Calculate the average of the function values
                                        // torch.mean handles tensors of any dimension, returning a scalar.
                                        double averageFunctionValue = torch.mean(functionValues).item<double>();

                                        // The estimated integral is the average function value multiplied by the integration volume
                                        double estimatedIntegral = averageFunctionValue * integrationVolume;

                                        return estimatedIntegral;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"IntegrateMonteCarlo (ND) caught an exception during computation: {ex.Message}{Environment.NewLine}{ex.StackTrace}");
                return 0.0;
            }
        }
    }
}