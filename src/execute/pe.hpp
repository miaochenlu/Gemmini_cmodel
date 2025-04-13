// pe.hpp - Processing Element for Gemmini Systolic Array using SPARTA framework
#pragma once

#include <cstdint>
#include <iostream>

#include "sparta/events/EventSet.hpp"
#include "sparta/events/UniqueEvent.hpp"
#include "sparta/log/MessageSource.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/ports/PortSet.hpp"
#include "sparta/ports/SignalPort.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/ResourceFactory.hpp"
#include "sparta/simulation/TreeNode.hpp"
#include "sparta/simulation/Unit.hpp"
#include "sparta/statistics/Counter.hpp"
#include "gemmini/common.hpp"
#include "utils/fifo.hpp"

BEGIN_NS(gemmini)

class PE;

// Parameter Set for PE
class PEParameterSet : public sparta::ParameterSet {
public:
    // Constructor - connect params to the PE's TreeNode
    PEParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) {
        // Parameters are initialized using the PARAMETER macro
    }

    // Parameters
    PARAMETER(uint32_t, compute_cycles, 0, "Cycles required for MAC operation")
    PARAMETER(uint32_t, act_width, 16, "Activation data width in bits")
    PARAMETER(uint32_t, weight_width, 16, "Weight data width in bits")
    PARAMETER(uint32_t, delay_cycles, 1, "Cycles of delay between connected PEs")
    PARAMETER(bool, debug_fifo, false, "Enable debug output for delay FIFOs")
};

// Port Set for PE
class PEPortSet : public sparta::PortSet {
public:
    // Input port structure
    struct PEInputPorts {
        sparta::DataInPort<int16_t> weight;       // Weight input (preloaded)
        sparta::DataInPort<int16_t> act;          // Activation input (from west)
        sparta::DataInPort<int32_t> partialSum;   // Partial sum input (from north)
        
        PEInputPorts(sparta::TreeNode* n) :
            weight(n, "inWeight", sparta::SchedulingPhase::Tick, 0),
            act(n, "inAct", sparta::SchedulingPhase::Tick, 0),
            partialSum(n, "inPartialSum", sparta::SchedulingPhase::Tick, 0) {}
    };
    
    // Output port structure
    struct PEOutputPorts {
        sparta::DataOutPort<int16_t> act;        // Activation output (to east)
        sparta::DataOutPort<int32_t> partialSum; // Partial sum output (to south)
        
        PEOutputPorts(sparta::TreeNode* n) :
            act(n, "outAct"),
            partialSum(n, "outPartialSum") {}
    };
    
    // Constructor
    PEPortSet(sparta::TreeNode* n)
        : sparta::PortSet(n),
          inputs(n),
          outputs(n) {
        // No need to register ports explicitly - the base class does this
    }

    // Input and output port structures
    PEInputPorts inputs;
    PEOutputPorts outputs;
};

// Processing Element class - basic compute unit for the systolic array
class PE : public sparta::Unit {
public:
    // Input data structure for PE
    struct PEInput {
        int16_t act = 0;      // Activation input
        int32_t psum = 0;     // Partial sum input
        bool act_valid = false;  // Is activation data valid
        bool psum_valid = false; // Is partial sum data valid
    };

    // Output data structure for PE
    struct PEOutput {
        int16_t act = 0;      // Activation output (to east)
        int32_t psum = 0;     // Partial sum output (to south)
        bool act_valid = false;  // Is activation data valid
        bool psum_valid = false; // Is partial sum data valid
    };

    // Static name for this resource
    static const char name[];

    // Constructor
    PE(sparta::TreeNode* node, const PEParameterSet* params);

    // Define parameter set type for use with ResourceFactory
    typedef PEParameterSet ParameterSet;

    // Factory for PE creation
    class Factory : public sparta::ResourceFactory<PE, PEParameterSet> {
    public:
        // Using parent constructor
        using sparta::ResourceFactory<PE, PEParameterSet>::ResourceFactory;
    };

    // Return port set
    PEPortSet & GetPortSet() { return mPortSet; }

    // Direct access methods
    void SetWeight(int16_t weight);
    void ReceiveActivation(int16_t act);
    void ReceivePartialSum(int32_t partialSum);

private:
    // Port set
    PEPortSet mPortSet;

    // Event set for scheduling
    sparta::EventSet mUnitEventSet;

    // Logger
    sparta::log::MessageSource mLogger;

    // Input and output state
    PEInput mInput;           // Current input data
    PEOutput mOutput;         // Current output data
    
    // Processing state
    int16_t mWeightReg = 0;       // Weight stored in PE
    int32_t mPartialSumReg = 0;   // Partial sum register (result)
    bool mBusy = false;           // Busy status
    uint32_t mCycleCounter = 0;   // Cycles remaining for computation
    
    // Configuration from parameters
    const uint32_t mComputeCycles;
    const uint32_t mActWidth;
    const uint32_t mWeightWidth;
    const uint32_t mDelayCycles;
    const bool mDebugFifo;

    // Statistics
    sparta::Counter mTotalMacs; // Count of MAC operations

    // Tick event for cycle-level computation
    sparta::UniqueEvent<> mTickEvent;
    
    // Delay FIFOs for modeling inter-PE delay
    std::unique_ptr<DelayFifo<int16_t>> mActDelayFifo;
    std::unique_ptr<DelayFifo<int32_t>> mPsumDelayFifo;

    // Internal methods
    void HandleWeight(const int16_t & weight);
    void HandleActivation(const int16_t & act);
    void HandlePartialSum(const int32_t & partialSum);
    void ComputeMAC();
    void Tick();
    
    // Check if we can perform computation
    bool CanCompute() const {
        return mInput.act_valid && mInput.psum_valid && !mBusy;
    }
};

END_NS(gemmini)