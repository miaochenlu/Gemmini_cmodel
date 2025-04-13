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
    PARAMETER(uint32_t, compute_time, 0, "Cycles required for MAC operation")
    PARAMETER(uint32_t, act_width, 16, "Activation data width in bits")
    PARAMETER(uint32_t, weight_width, 16, "Weight data width in bits")
};

// Port Set for PE
class PEPortSet : public sparta::PortSet {
public:
    // Constructor
    PEPortSet(sparta::TreeNode* n)
        : sparta::PortSet(n),
          // Data input ports
          inWeight(n, "inWeight", sparta::SchedulingPhase::Tick, 0),
          inAct(n, "inAct", sparta::SchedulingPhase::Tick, 0),
          inWtPs(n, "inWtPs", sparta::SchedulingPhase::Tick, 0),
          inWtValid(n, "inWtValid", sparta::SchedulingPhase::Tick, 0),
          // Data output ports
          outAct(n, "outAct"), 
          outWtPs(n, "outWtPs") {
        // No need to register ports explicitly - the base class does this
    }

    // Input ports - data arrives to these ports from outside
    sparta::DataInPort<int16_t> inWeight;  // Direct weight input (for initialization)
    sparta::DataInPort<int16_t> inAct;     // Activation input
    sparta::DataInPort<int32_t> inWtPs;    // Weight/Partial Sum input
    sparta::DataInPort<uint32_t> inWtValid; // Weight valid signal (1=weight loading mode)

    // Output ports - data sent from these ports to outside
    sparta::DataOutPort<int16_t> outAct;   // Activation output
    sparta::DataOutPort<int32_t> outWtPs;  // Weight/Partial Sum output
};

// Processing Element class - basic compute unit for the systolic array
class PE : public sparta::Unit {
public:
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
    void ReceiveWeightPartialSum(int32_t wtPs);
    void SetWeightValidSignal(uint32_t valid);

private:
    // Port set
    PEPortSet mPortSet;

    // Event set for scheduling
    sparta::EventSet mUnitEventSet;

    // Logger
    sparta::log::MessageSource mLogger;

    // Internal state
    int16_t mWeightReg = 0;      // Weight stored in PE
    int16_t mActReg = 0;         // Activation register
    int32_t mPartialSumReg = 0;  // Partial sum register
    bool mBusy = false;          // Busy status
    bool mWeightLoadingMode = false; // Weight loading mode flag
    uint32_t mCycleCounter = 0;  // Cycles remaining for computation

    // Configuration from parameters
    const uint32_t mComputeTime;
    const uint32_t mActWidth;
    const uint32_t mWeightWidth;

    // Statistics
    sparta::Counter mTotalMacs; // Count of MAC operations

    // Tick event
    sparta::UniqueEvent<> mTickEvent;

    // Internal methods
    void HandleWeight(const int16_t & weight);
    void HandleActivation(const int16_t & act);
    void HandleWeightPartialSum(const int32_t & wtPs);
    void HandleWeightValid(const uint32_t & valid);
    void ProcessData();
    void Tick();
};

END_NS(gemmini)