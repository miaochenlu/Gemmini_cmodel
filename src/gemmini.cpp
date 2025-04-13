// gemmini.cpp - Implementation of top-level Gemmini simulator using SPARTA
#include "gemmini/gemmini.hpp"
#include "sparta/kernel/Scheduler.hpp"
#include <iostream>

namespace gemmini {
// GemminiSimulation Constructor
GemminiSimulation::GemminiSimulation(sparta::Scheduler* scheduler)
    : sparta::app::Simulation("GemminiSim", scheduler) {
    // Nothing to do here - resource construction happens in buildTree_()
}

// Destructor - ensure proper cleanup
GemminiSimulation::~GemminiSimulation() { getRoot()->enterTeardown(); }

// Implementation of virtual methods from Simulation
void GemminiSimulation::buildTree_() {
    // Create resources
    auto rootNode = getRoot();

    // Create matrix multiplier node
    sparta::TreeNode* mmNode =
        new sparta::TreeNode(rootNode, "matrix_multiplier", "Matrix Multiplier");

    // Create parameter set for matrix multiplier
    auto mmParams = new MatrixMultiplier::ParameterSet(mmNode);

    // No need to manually set parameters - they have default values

    // Create matrix multiplier resource
    MatrixMultiplier::Factory mmFactory;
    mMatrixMultiplier = static_cast<MatrixMultiplier*>(mmFactory.createResource(mmNode, mmParams));
}

void GemminiSimulation::configureTree_() {
    // Configure the simulation tree if needed
    // In this simple implementation, we don't need additional configuration
}

void GemminiSimulation::bindTree_() {
    // Bind the simulation tree components if needed
    // In this simple implementation, all binding happens in component constructors
}

// Run simulation with input matrices
void GemminiSimulation::RunSimulation(const MatrixPtr & matrixA, const MatrixPtr & matrixB) {
    std::cout << "Starting Gemmini matrix multiplication simulation..." << std::endl;

    // Print matrix dimensions
    std::cout << "Matrix A: " << matrixA->Rows() << "x" << matrixA->Cols() << std::endl;
    std::cout << "Matrix B: " << matrixB->Rows() << "x" << matrixB->Cols() << std::endl;

    // Calculate expected cycles
    // For a matrix multiplication using a 4x4 systolic array:
    // - Setup: 1 cycle
    // - Systolic array processing: rows + cols - 1 + compute_cycles cycles
    // - Computation complete: 1 cycle
    // - Block handling: number of blocks * (above)
    uint32_t rowsPerBlock = 4;
    uint32_t colsPerBlock = 4;
    uint32_t computeTime = 0; // Changed to 0 to match updated design

    uint32_t rowBlocks = (matrixA->Rows() + rowsPerBlock - 1) / rowsPerBlock;
    uint32_t colBlocks = (matrixB->Cols() + colsPerBlock - 1) / colsPerBlock;
    uint32_t cyclesPerBlock = 1 + rowsPerBlock + colsPerBlock - 1 + computeTime + 1;
    uint32_t expectedCycles = rowBlocks * colBlocks * cyclesPerBlock;

    // Add extra cycles for setup and final result handling
    expectedCycles += 10;

    std::cout << "Expected simulation time: " << expectedCycles << " cycles" << std::endl;

    // Perform matrix multiplication
    mMatrixMultiplier->Multiply(matrixA, matrixB);

    // Run simulation for expected time
    runRaw(expectedCycles);

    // Get results
    MatrixPtr result = mMatrixMultiplier->GetResult();

    // Print results
    std::cout << "Matrix multiplication result:" << std::endl;
    std::cout << *result << std::endl;
}

} // namespace gemmini