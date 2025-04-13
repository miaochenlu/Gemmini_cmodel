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

#include "utils/common.hpp"
#include "execute/matrix.hpp"
#include "execute/systolic_array.hpp"

BEGIN_NS(gemmini)

class MatrixMultiplier;

// Parameter Set for MatrixMultiplier
class MatrixMultiplierParameterSet : public sparta::ParameterSet {
public:
    // Constructor - connect params to the MatrixMultiplier's TreeNode
    MatrixMultiplierParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) {
        // Parameters are initialized using the PARAMETER macro
    }

    // Parameters
    PARAMETER(uint32_t, systolic_rows, 4, "Number of rows in systolic array")
    PARAMETER(uint32_t, systolic_cols, 4, "Number of columns in systolic array")
};

// Port Set for MatrixMultiplier
class MatrixMultiplierPortSet : public sparta::PortSet {
public:
    // Constructor
    MatrixMultiplierPortSet(sparta::TreeNode* n)
        : sparta::PortSet(n),

          // Declare input ports with correct signatures
          in_matrix_a(n, "in_matrix_a", sparta::SchedulingPhase::Tick, 0),
          in_matrix_b(n, "in_matrix_b", sparta::SchedulingPhase::Tick, 0),
          in_control(n, "in_control", sparta::SchedulingPhase::Tick, 0),

          // Declare output ports with correct signatures
          out_result(n, "out_result") {
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
class MatrixMultiplier : public sparta::Unit {
public:
    // Static name for this resource
    static const char name[];

    // Constructor
    MatrixMultiplier(sparta::TreeNode* node, const MatrixMultiplierParameterSet* params);

    // Define parameter set type for use with ResourceFactory
    typedef MatrixMultiplierParameterSet ParameterSet;

    // Factory for MatrixMultiplier creation
    class Factory : public sparta::ResourceFactory<MatrixMultiplier, MatrixMultiplierParameterSet> {
    public:
        // Using parent constructor
        using sparta::ResourceFactory<MatrixMultiplier,
                                      MatrixMultiplierParameterSet>::ResourceFactory;
    };

    // Direct access method for simulation
    void Multiply(const MatrixPtr & a, const MatrixPtr & b);

    MatrixPtr GetResult() const { return mResultMatrix; }

private:
    // Port set
    MatrixMultiplierPortSet mPortSet;

    // Ports to/from Systolic Array
    sparta::DataOutPort<MatrixPtr> mToSystolicWeights;
    sparta::DataOutPort<VectorPtr> mToSystolicVector;
    sparta::DataInPort<MatrixPtr> mFromSystolicResults;

    // Event set for scheduling
    sparta::EventSet mUnitEventSet;

    // Logger
    sparta::log::MessageSource mLogger;

    // Configuration
    const uint32_t mSystolicRows;
    const uint32_t mSystolicCols;

    // Current state
    bool mBusy = false;
    uint32_t mCurrentRowBlock = 0;
    uint32_t mCurrentColBlock = 0;
    uint32_t mTotalRowBlocks = 0;
    uint32_t mTotalColBlocks = 0;
    MatrixPtr mMatrixA;
    MatrixPtr mMatrixB;
    MatrixPtr mResultMatrix;

    // Statistics
    sparta::Counter mTotalMms;    // Count of matrix multiplications
    sparta::Counter mTotalBlocks; // Count of block operations

    // Internal methods
    void HandleMatrixA(const MatrixPtr & a);
    void HandleMatrixB(const MatrixPtr & b);
    void HandleControl(const uint32_t & signal);
    void HandleSystolicResults(const MatrixPtr & results);

    void StartMultiplication();
    void ProcessNextBlock();
    void MultiplierDone();
};

END_NS(gemmini)