// gemmini.cpp - Implementation of top-level Gemmini simulator using SPARTA
#include "gemmini/gemmini.hpp"
#include "sparta/kernel/Scheduler.hpp"
#include <iostream>

namespace gemmini
{
    // GemminiSimulation Constructor
    GemminiSimulation::GemminiSimulation(sparta::Scheduler* scheduler) :
        sparta::app::Simulation("GemminiSim", scheduler)
    {
        // Nothing to do here - resource construction happens in buildTree_()
    }
    
    // Destructor - ensure proper cleanup
    GemminiSimulation::~GemminiSimulation() {
        getRoot()->enterTeardown();
    }
    
    // Implementation of virtual methods from Simulation
    void GemminiSimulation::buildTree_() {
        // Create resources
        auto root_node = getRoot();
        
        // Create matrix multiplier node
        sparta::TreeNode* mm_node = new sparta::TreeNode(root_node, "matrix_multiplier", "Matrix Multiplier");
        
        // Create parameter set for matrix multiplier
        auto mm_params = new MatrixMultiplier::ParameterSet(mm_node);
        
        // No need to manually set parameters - they have default values
        
        // Create matrix multiplier resource
        MatrixMultiplier::Factory mm_factory;
        matrix_multiplier_ = static_cast<MatrixMultiplier*>(mm_factory.createResource(mm_node, mm_params));
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
    void GemminiSimulation::runSimulation(const MatrixPtr& matrix_a, const MatrixPtr& matrix_b) {
        std::cout << "Starting Gemmini matrix multiplication simulation..." << std::endl;
        
        // Print matrix dimensions
        std::cout << "Matrix A: " << matrix_a->rows() << "x" << matrix_a->cols() << std::endl;
        std::cout << "Matrix B: " << matrix_b->rows() << "x" << matrix_b->cols() << std::endl;
        
        // Calculate expected cycles
        // For a matrix multiplication using a 4x4 systolic array:
        // - Setup: 1 cycle
        // - Systolic array processing: rows + cols - 1 + compute_time cycles
        // - Computation complete: 1 cycle
        // - Block handling: number of blocks * (above)
        uint32_t rows_per_block = 4;
        uint32_t cols_per_block = 4;
        uint32_t compute_time = 1;
        
        uint32_t row_blocks = (matrix_a->rows() + rows_per_block - 1) / rows_per_block;
        uint32_t col_blocks = (matrix_b->cols() + cols_per_block - 1) / cols_per_block;
        uint32_t cycles_per_block = 1 + rows_per_block + cols_per_block - 1 + compute_time + 1;
        uint32_t expected_cycles = row_blocks * col_blocks * cycles_per_block;
        
        // Add extra cycles for setup and final result handling
        expected_cycles += 10;
        
        std::cout << "Expected simulation time: " << expected_cycles << " cycles" << std::endl;
        
        // Perform matrix multiplication
        matrix_multiplier_->multiply(matrix_a, matrix_b);
        
        // Run simulation for expected time
        runRaw(expected_cycles);
        
        // Get results
        MatrixPtr result = matrix_multiplier_->getResult();
        
        // Print results
        std::cout << "Matrix multiplication result:" << std::endl;
        std::cout << *result << std::endl;
    }
    
} // namespace gemmini 