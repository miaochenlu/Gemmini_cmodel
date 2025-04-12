// matrix_multiplier.hpp - Matrix Multiplier for Gemmini using SPARTA framework
#pragma once

#include <cstdint>
#include <vector>
#include <memory>

#include "sparta/ports/PortSet.hpp"
#include "sparta/ports/SignalPort.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/events/EventSet.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/TreeNode.hpp"
#include "sparta/simulation/ResourceFactory.hpp"
#include "sparta/statistics/Counter.hpp"
#include "sparta/log/MessageSource.hpp"

#include "gemmini/matrix.hpp"
#include "gemmini/systolic_array.hpp"

namespace gemmini
{
    class MatrixMultiplier;

    // Parameter Set for MatrixMultiplier
    class MatrixMultiplierParameterSet : public sparta::ParameterSet
    {
    public:
        // Constructor - connect params to the MatrixMultiplier's TreeNode
        MatrixMultiplierParameterSet(sparta::TreeNode* n) :
            sparta::ParameterSet(n)
        {
            // Parameters are initialized using the PARAMETER macro
        }

        // Parameters
        PARAMETER(uint32_t, systolic_rows, 4, "Number of rows in systolic array")
        PARAMETER(uint32_t, systolic_cols, 4, "Number of columns in systolic array")
    };

    // Port Set for MatrixMultiplier
    class MatrixMultiplierPortSet : public sparta::PortSet
    {
    public:
        // Constructor
        MatrixMultiplierPortSet(sparta::TreeNode* n) :
            sparta::PortSet(n),
            
            // Declare input ports with correct signatures
            in_matrix_a(n, "in_matrix_a", sparta::SchedulingPhase::Tick, 0),
            in_matrix_b(n, "in_matrix_b", sparta::SchedulingPhase::Tick, 0),
            in_control(n, "in_control", sparta::SchedulingPhase::Tick, 0),
            
            // Declare output ports with correct signatures
            out_result(n, "out_result")
        {
            // No need to register ports explicitly - the base class does this
        }

        // Input ports
        sparta::DataInPort<MatrixPtr> in_matrix_a;
        sparta::DataInPort<MatrixPtr> in_matrix_b;
        sparta::DataInPort<uint32_t> in_control;
        
        // Output ports
        sparta::DataOutPort<MatrixPtr> out_result;
    };

    // Matrix Multiplier class - Orchestrates the systolic array to perform matrix multiplication
    class MatrixMultiplier : public sparta::Unit
    {
    public:
        // Static name for this resource
        static const char name[];
        
        // Constructor
        MatrixMultiplier(sparta::TreeNode* node, const MatrixMultiplierParameterSet* params);
        
        // Define parameter set type for use with ResourceFactory
        typedef MatrixMultiplierParameterSet ParameterSet;
        
        // Factory for MatrixMultiplier creation
        class Factory : public sparta::ResourceFactory<MatrixMultiplier, MatrixMultiplierParameterSet>
        {
        public:
            // Using parent constructor
            using sparta::ResourceFactory<MatrixMultiplier, MatrixMultiplierParameterSet>::ResourceFactory;
        };
        
        // Direct access method for simulation
        void multiply(const MatrixPtr& a, const MatrixPtr& b);
        MatrixPtr getResult() const { return result_matrix_; }
        
    private:
        // Port set
        MatrixMultiplierPortSet port_set_;
        
        // Ports to/from Systolic Array
        sparta::DataOutPort<MatrixPtr> to_systolic_weights_;
        sparta::DataOutPort<VectorPtr> to_systolic_vector_;
        sparta::DataInPort<MatrixPtr> from_systolic_results_;
        
        // Event set for scheduling
        sparta::EventSet unit_event_set_;
        
        // Logger
        sparta::log::MessageSource logger_;
        
        // Configuration
        const uint32_t systolic_rows_;
        const uint32_t systolic_cols_;
        
        // Current state
        bool busy_ = false;
        uint32_t current_row_block_ = 0;
        uint32_t current_col_block_ = 0;
        uint32_t total_row_blocks_ = 0;
        uint32_t total_col_blocks_ = 0;
        MatrixPtr matrix_a_;
        MatrixPtr matrix_b_;
        MatrixPtr result_matrix_;
        
        // Statistics
        sparta::Counter total_mms_; // Count of matrix multiplications
        sparta::Counter total_blocks_; // Count of block operations
        
        // Internal methods
        void handleMatrixA_(const MatrixPtr& a);
        void handleMatrixB_(const MatrixPtr& b);
        void handleControl_(const uint32_t& signal);
        void handleSystolicResults_(const MatrixPtr& results);
        
        void startMultiplication_();
        void processNextBlock_();
        void multiplierDone_();
    };
    
} // namespace gemmini 