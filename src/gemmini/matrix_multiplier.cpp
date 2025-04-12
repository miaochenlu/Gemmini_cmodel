// matrix_multiplier.cpp - Implementation of Matrix Multiplier for Gemmini using SPARTA
#include "gemmini/matrix_multiplier.hpp"
#include "sparta/kernel/Scheduler.hpp"
#include "sparta/kernel/SpartaHandler.hpp"
#include <algorithm>
#include <iostream>

namespace gemmini
{
    // Initialize static name
    const char MatrixMultiplier::name[] = "matrix_multiplier";

    // MatrixMultiplier Constructor
    MatrixMultiplier::MatrixMultiplier(sparta::TreeNode* node, const MatrixMultiplierParameterSet* params) :
        sparta::Unit(node),
        port_set_(node),
        to_systolic_weights_(node, "to_systolic_weights"),
        to_systolic_vector_(node, "to_systolic_vector"),
        from_systolic_results_(node, "from_systolic_results", sparta::SchedulingPhase::Tick, 0),
        unit_event_set_(node),
        logger_(node, "matrix_multiplier", "Matrix Multiplier Log"),
        systolic_rows_(params->systolic_rows),
        systolic_cols_(params->systolic_cols),
        total_mms_(getStatisticSet(), "total_mms", "Count of matrix multiplications", sparta::Counter::COUNT_NORMAL),
        total_blocks_(getStatisticSet(), "total_blocks", "Count of block operations", sparta::Counter::COUNT_NORMAL)
    {
        // Register port handlers
        port_set_.in_matrix_a.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(MatrixMultiplier, handleMatrixA_, MatrixPtr));
        port_set_.in_matrix_b.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(MatrixMultiplier, handleMatrixB_, MatrixPtr));
        port_set_.in_control.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(MatrixMultiplier, handleControl_, uint32_t));
        from_systolic_results_.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(MatrixMultiplier, handleSystolicResults_, MatrixPtr));
        
        // Create systolic array child
        sparta::TreeNode* systolic_node = new sparta::TreeNode(node, "systolic_array", "Systolic Array");
        
        // Create systolic array parameter set
        auto params_for_systolic = new SystolicArrayParameterSet(systolic_node);
        
        // No need to manually set parameters - they have default values
        
        // Create systolic array resource
        SystolicArray::Factory systolic_factory;
        SystolicArray* systolic_array = static_cast<SystolicArray*>(systolic_factory.createResource(systolic_node, params_for_systolic));
        
        // Connect ports
        to_systolic_weights_.bind(systolic_array->getPortSet().in_weights);
        to_systolic_vector_.bind(systolic_array->getPortSet().in_vector);
        from_systolic_results_.bind(systolic_array->getPortSet().out_results);
    }

    // Handle receiving matrix A
    void MatrixMultiplier::handleMatrixA_(const MatrixPtr& a)
    {
        // Avoid using logger_.debug() or similar
        #ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Received matrix A: " << a->rows() << "x" << a->cols() << std::endl;
        #endif
        matrix_a_ = a;
    }
    
    // Handle receiving matrix B
    void MatrixMultiplier::handleMatrixB_(const MatrixPtr& b)
    {
        #ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Received matrix B: " << b->rows() << "x" << b->cols() << std::endl;
        #endif
        matrix_b_ = b;
    }
    
    // Handle control signals
    void MatrixMultiplier::handleControl_(const uint32_t& signal)
    {
        #ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Received control signal: " << signal << std::endl;
        #endif
        
        // If it's a start signal, start matrix multiplication
        if (signal == 1 && matrix_a_ && matrix_b_) {
            startMultiplication_();
        }
    }

    // Multiply two matrices - direct method for simulation
    void MatrixMultiplier::multiply(const MatrixPtr& a, const MatrixPtr& b)
    {
        // Pass matrices to the port handlers
        handleMatrixA_(a);
        handleMatrixB_(b);
        
        // Start multiplication process
        startMultiplication_();
    }

    // Start the matrix multiplication process
    void MatrixMultiplier::startMultiplication_()
    {
        if (busy_) {
            #ifdef DEBUG_MATRIX_MULTIPLIER
            std::cout << "Matrix multiplier is busy, ignoring new multiplication request" << std::endl;
            #endif
            return;
        }
        
        // Check compatibility of matrices
        if (matrix_a_->cols() != matrix_b_->rows()) {
            std::cerr << "Matrix dimensions incompatible for multiplication: " 
                     << matrix_a_->rows() << "x" << matrix_a_->cols() << " * "
                     << matrix_b_->rows() << "x" << matrix_b_->cols() << std::endl;
            return;
        }
        
        #ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Starting matrix multiplication: " 
                 << matrix_a_->rows() << "x" << matrix_a_->cols() << " * "
                 << matrix_b_->rows() << "x" << matrix_b_->cols() << std::endl;
        #endif
        
        // Set busy flag
        busy_ = true;
        
        // Reset block counters
        current_row_block_ = 0;
        current_col_block_ = 0;
        
        // Calculate number of blocks needed
        total_row_blocks_ = (matrix_a_->rows() + systolic_rows_ - 1) / systolic_rows_;
        total_col_blocks_ = (matrix_b_->cols() + systolic_cols_ - 1) / systolic_cols_;
        
        // Initialize result matrix
        result_matrix_ = create_matrix_ptr<Matrix>(matrix_a_->rows(), matrix_b_->cols());
        
        // Start processing the first block
        processNextBlock_();
        
        // Update statistics
        total_mms_++;
    }

    // Process next block in the matrix multiplication
    void MatrixMultiplier::processNextBlock_()
    {
        // Calculate block dimensions and offsets
        uint32_t row_offset = current_row_block_ * systolic_rows_;
        uint32_t col_offset = current_col_block_ * systolic_cols_;
        
        uint32_t block_rows = std::min(systolic_rows_, matrix_a_->rows() - row_offset);
        uint32_t block_cols = std::min(systolic_cols_, matrix_b_->cols() - col_offset);
        
        #ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Processing block [" << current_row_block_ << "," << current_col_block_ << "]: "
                 << block_rows << "x" << block_cols << std::endl;
        #endif
        
        // Create weight matrix for this block (transposed portion of B)
        MatrixPtr weights = create_matrix_ptr<Matrix>(systolic_rows_, systolic_cols_);
        
        // Fill weight matrix with transposed values from B
        for (uint32_t r = 0; r < block_rows; ++r) {
            for (uint32_t c = 0; c < block_cols; ++c) {
                if (row_offset + r < matrix_a_->rows() && c < matrix_b_->rows()) {
                    // Transpose during load
                    weights->at(r, c) = matrix_b_->at(c, col_offset + r);
                }
            }
        }
        
        // Send weights to systolic array
        to_systolic_weights_.send(weights);
        
        // Create input vectors for this block (from A)
        std::vector<VectorPtr> input_vectors;
        
        // For each row in the block from A
        for (uint32_t r = 0; r < block_rows; ++r) {
            VectorPtr row_vector = create_matrix_ptr<Vector>(matrix_a_->cols());
            
            // Fill vector with values from A
            for (uint32_t c = 0; c < matrix_a_->cols(); ++c) {
                if (row_offset + r < matrix_a_->rows()) {
                    (*row_vector)[c] = matrix_a_->at(row_offset + r, c);
                }
            }
            
            input_vectors.push_back(row_vector);
        }
        
        // Process each input vector
        for (const auto& vector : input_vectors) {
            // Send vector to systolic array
            to_systolic_vector_.send(vector);
        }
        
        // Update statistics
        total_blocks_++;
    }

    // Handle results from systolic array
    void MatrixMultiplier::handleSystolicResults_(const MatrixPtr& results)
    {
        if (!busy_) {
            #ifdef DEBUG_MATRIX_MULTIPLIER
            std::cout << "Received results when not busy, ignoring" << std::endl;
            #endif
            return;
        }
        
        #ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Received results for block [" << current_row_block_ << "," << current_col_block_ << "]" << std::endl;
        #endif
        
        // Copy block results to the appropriate position in the final result matrix
        uint32_t row_offset = current_row_block_ * systolic_rows_;
        uint32_t col_offset = current_col_block_ * systolic_cols_;
        
        uint32_t block_rows = std::min(systolic_rows_, matrix_a_->rows() - row_offset);
        uint32_t block_cols = std::min(systolic_cols_, matrix_b_->cols() - col_offset);
        
        // Copy results
        for (uint32_t r = 0; r < block_rows; ++r) {
            for (uint32_t c = 0; c < block_cols; ++c) {
                result_matrix_->at(row_offset + r, col_offset + c) = results->at(r, c);
            }
        }
        
        // Move to next block
        current_col_block_++;
        if (current_col_block_ >= total_col_blocks_) {
            current_col_block_ = 0;
            current_row_block_++;
            
            if (current_row_block_ >= total_row_blocks_) {
                // All blocks processed
                multiplierDone_();
                return;
            }
        }
        
        // Process next block
        processNextBlock_();
    }

    // Called when all blocks have been processed
    void MatrixMultiplier::multiplierDone_()
    {
        #ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Matrix multiplication complete" << std::endl;
        #endif
        
        // Send result to output port
        port_set_.out_result.send(result_matrix_);
        
        // Reset busy flag
        busy_ = false;
    }
    
} // namespace gemmini 