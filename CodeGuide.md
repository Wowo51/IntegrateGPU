# IntegrateGPU Code Guide

## Overview

**IntegrateGPU** is a C# library built on top of [TorchSharp](https://github.com/dotnet/TorchSharp) that performs Monte Carlo numerical integration on the GPU or CPU using tensors. It supports both 1-dimensional and N-dimensional definite integrals of arbitrary mathematical functions.

It includes:

* A main library project (`IntegrateGPU`)
* A unit testing project (`IntegrateGPUTest`) using MSTest
* TorchSharp for tensor operations and CUDA GPU acceleration (when available)

---

## Installation & Setup

### Prerequisites:

* .NET 9 SDK
* Visual Studio 2022 or later with .NET 9 support
* TorchSharp 0.105.0

### Projects:

* `IntegrateGPU`: The main integration library
* `IntegrateGPUTest`: Contains over 40 MSTest-based unit tests

### Dependencies:

```xml
<PackageReference Include="TorchSharp-cuda-windows" Version="0.105.0" />
```

If you don't have a CUDA-enabled GPU, TorchSharp will fall back to CPU.

---

## Using the Integration API

### 1D Monte Carlo Integration

```csharp
double result = IntegrateGPU.IntegrateMonteCarlo(
    x => torch.pow(x, 2.0), // f(x) = x^2
    0.0,                    // lower bound
    1.0,                    // upper bound
    1_000_000,             // number of samples
    torch.CPU              // or torch.CUDA if available
);
```

### N-Dimensional Integration

```csharp
double result = IntegrateGPU.IntegrateMonteCarlo(
    samples => samples.index(Ellipsis, 0).add(samples.index(Ellipsis, 1)),
    torch.tensor(new double[] { 0.0, 0.0 }, dtype: float64, device: torch.CPU),
    torch.tensor(new double[] { 1.0, 1.0 }, dtype: float64, device: torch.CPU),
    2_000_000,
    torch.CPU
);
```

---

## How It Works

### Monte Carlo Integration Principle

1. Randomly sample points within the bounds.
2. Evaluate the function at each sample.
3. Multiply the average value by the total integration volume.

### GPU Acceleration

TorchSharp tensors and operations (such as `torch.rand`, `torch.mean`) are dispatched to the GPU when `torch.CUDA` is selected. This enables highly parallel execution.

---

## API Reference

### `IntegrateMonteCarlo(Func<Tensor, Tensor>, double, double, long, Device)`

* Converts scalar bounds to 1D tensors
* Calls the N-dimensional method with a single-dimension tensor

### `IntegrateMonteCarlo(Func<Tensor, Tensor>, Tensor, Tensor, long, Device)`

* Assumes N-dimensional input
* Generates uniform random samples within a hypercube
* Evaluates user-provided function
* Computes volume and final integral estimate

---

## Testing

### Structure

Each test in `IntegrateGPUTest` defines:

* A mathematical function as a `Func<Tensor, Tensor>`
* Known analytical bounds and expected result
* A high number of samples (1–2 million) for Monte Carlo accuracy

### Example Test

```csharp
Func<Tensor, Tensor> func = x => torch.sin(x);
double expected = 2.0;
Assert.AreEqual(expected, result, tolerance);
```

### Test Coverage

* 1D Functions: polynomials, exponential, trigonometric
* N-D Functions: sum, product, constant, composite
* Dimensions: 1D, 2D, 3D, 4D, 5D

---

## Performance Tuning Tips

* **Use CUDA** (`torch.CUDA`) for faster evaluation on large sample sizes
* **Increase samples** to improve accuracy at the cost of performance
* **Dispose tensors** if you're allocating many in a loop

---

## Error Handling

* All API methods catch exceptions and return `0.0` if errors occur
* Detailed messages are logged to `Console.Error`
* Common errors: mismatched bounds, empty tensors, invalid inputs

---

## Extending the Library

You can:

* Add **importance sampling** for better convergence
* Implement **stratified sampling**
* Add **parallelism** using TorchSharp batch operations
* Support **probabilistic functions** with expected-value Monte Carlo methods

---

## Summary

**IntegrateGPU** is a GPU-accelerated C# numerical integration library using Monte Carlo sampling. With robust test coverage and TorchSharp integration, it is ideal for fast approximations of definite integrals in one or more dimensions, especially on high-performance hardware.

For production or research workloads, it offers a clean and scalable architecture to build upon.

Copyright [TranscendAI.tech](https://TranscendAI.tech) 2025.
Authored by Warren Harding. AI assisted.