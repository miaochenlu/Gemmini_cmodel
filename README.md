## Data Flow in the Systolic Array

The key to systolic array operation is the skewed scheduling that creates a diagonal wavefront pattern:

1. In cycle 0, data enters the leftmost PE in the top row
2. In cycle 1, data enters the second PE in the top row and the leftmost PE in the second row
3. This diagonal pattern continues, with data flowing from left to right
4. Each PE performs a MAC operation when it receives data
5. The result of the matrix-vector multiplication is found in the rightmost column of PEs

This implementation offers several advantages:

- Parallel computation across multiple PEs
- Efficient data reuse (weights are stationary)
- Reduced global data movement (data flows locally between PEs)
- Scalable design (can be extended to larger arrays)

The PE and systolic array implementation for Gemmini successfully demonstrates the fundamental concepts of systolic array architecture used in modern ML accelerators.

## Gemmini Systolic Array Simulator

A cycle-accurate simulator for the Gemmini systolic array accelerator using the SPARTA framework.

### Building the Project

```bash
mkdir -p build && cd build
cmake ..
make
```

### Running the Tests

The project includes test code for verifying the functionality of the Gemmini systolic array:

```bash
# Run with normal output
./bin/gemmini_simulator

# Run with verbose output
./bin/gemmini_simulator -v

# Show help
./bin/gemmini_simulator -h
```

### Testing the Gemmini Systolic Array

The test code demonstrates:

1. **Matrix and Vector Creation**: Random matrices and vectors are generated for testing, ensuring they're properly sized for the 4x4 systolic array.

2. **Matrix-Vector Multiplication**: Tests demonstrate how vectors are multiplied by weight matrices, which is the fundamental operation in the systolic array design.

3. **Matrix-Matrix Multiplication**: The full matrix multiplication test demonstrates how larger matrix calculations can be broken down into operations suitable for the systolic array.

4. **Result Verification**: Expected results are calculated and compared with the actual results from the systolic array operations.

The PE (Processing Element) implementation in this project supports:

- Weight loading for stationary weights
- Data streaming through the array
- MAC (Multiply-Accumulate) operations
- Result accumulation

The 4x4 systolic array demonstrates the data flow pattern, where:

- Weights are loaded into each PE
- Input data is fed into the leftmost column
- Data propagates horizontally through the array
- Results accumulate in the PEs and are read from the rightmost column

This design effectively demonstrates the principles of systolic array architectures used in ML accelerators.
