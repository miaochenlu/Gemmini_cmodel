// systolic_array.hpp - Systolic Array for Gemmini using SPARTA framework
#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

#include "sparta/ports/PortSet.hpp"
#include "sparta/ports/SignalPort.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/events/EventSet.hpp"
#include "sparta/events/UniqueEvent.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/TreeNode.hpp"
#include "sparta/simulation/ResourceFactory.hpp"
#include "sparta/log/MessageSource.hpp"
#include "sparta/statistics/Counter.hpp"

#include "gemmini/common.hpp"
#include "gemmini/matrix.hpp"
#include "gemmini/pe.hpp"

BEGIN_NS(gemmini)

class SystolicArray;

// Parameter Set for SystolicArray
class SystolicArrayParameterSet : public sparta::ParameterSet {
public:
    // Constructor - connect params to the SystolicArray's TreeNode
    SystolicArrayParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) {
        // Parameters are initialized using the PARAMETER macro
    }

    // Parameters
    PARAMETER(uint32_t, rows, 4, "Number of rows in systolic array")
    PARAMETER(uint32_t, cols, 4, "Number of columns in systolic array")
    PARAMETER(uint32_t, compute_cycles, 0, "Cycles required for PE MAC operation")
};

// Port Set for SystolicArray
class SystolicArrayPortSet : public sparta::PortSet {
public:
    // Constructor
    SystolicArrayPortSet(sparta::TreeNode* n)
        : sparta::PortSet(n),

          // Declare input ports with correct signature
          in_weights(n, "in_weights", sparta::SchedulingPhase::Tick, 0),
          in_vector(n, "in_vector", sparta::SchedulingPhase::Tick, 0),
          in_control(n, "in_control", sparta::SchedulingPhase::Tick, 0),

          // Declare output ports with correct signature
          out_results(n, "out_results") {
        // No need to register ports explicitly - the base class does this
    }

    // Input ports
    sparta::DataInPort<MatrixPtr> in_weights;
    sparta::DataInPort<VectorPtr> in_vector;
    sparta::DataInPort<uint32_t> in_control;

    // Output ports
    sparta::DataOutPort<MatrixPtr> out_results;
};

// Systolic Array class - 2D array of Processing Elements using SPARTA
class SystolicArray : public sparta::Unit {
public:
    // Static name for this resource
    static const char name[];

    // Constructor
    SystolicArray(sparta::TreeNode* node, const SystolicArrayParameterSet* params);

    // Define parameter set type for use with ResourceFactory
    typedef SystolicArrayParameterSet ParameterSet;

    // Factory for SystolicArray creation
    class Factory : public sparta::ResourceFactory<SystolicArray, SystolicArrayParameterSet> {
    public:
        // Using parent constructor
        using sparta::ResourceFactory<SystolicArray, SystolicArrayParameterSet>::ResourceFactory;
    };

    // Return port set
    SystolicArrayPortSet & GetPortSet() { return mPortSet; }

private:
    // Port set
    SystolicArrayPortSet mPortSet;

    // Event set for scheduling
    sparta::EventSet mUnitEventSet;

    // Logger
    sparta::log::MessageSource mLogger;

    // Configuration parameters
    const uint32_t mRows;
    const uint32_t mCols;
    const uint32_t mComputeCycles;

    // Array of Processing Elements
    std::vector<PE*> mPEs; // Flattened 2D array for easier access

    // Current state
    bool mProcessing = false;
    uint32_t mCurrentCycle = 0;
    uint32_t mTotalCyclesNeeded = 0;
    VectorPtr mCurrentInput;
    MatrixPtr mResultMatrix;

    // Statistics
    sparta::Counter mTotalMatrixOps; // Count of matrix operations

    // Tick event
    sparta::UniqueEvent<> mTickEvent;

    // Internal methods
    void HandleWeights(const MatrixPtr & weights);
    void HandleVector(const VectorPtr & input);
    void HandleControl(const uint32_t & signal);

    void ProcessOneCycle();
    void ComputationComplete();
    void Tick();

    // Helper methods
    PE* GetPE(uint32_t row, uint32_t col) { return mPEs[row * mCols + col]; }

    std::string GetPEName(uint32_t row, uint32_t col) const {
        return "pe_" + std::to_string(row) + "_" + std::to_string(col);
    }
};

END_NS(gemmini)