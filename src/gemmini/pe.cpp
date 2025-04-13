// pe.cpp - Implementation of Processing Element for Gemmini Systolic Array using SPARTA
#include "gemmini/pe.hpp"
#include "sparta/events/StartupEvent.hpp"
#include "sparta/kernel/Scheduler.hpp"
#include "sparta/kernel/SpartaHandler.hpp"
#include <iostream>

namespace gemmini {
// Initialize static name
const char PE::name[] = "pe";

// PE Constructor
PE::PE(sparta::TreeNode* node, const PEParameterSet* params)
    : sparta::Unit(node), mPortSet(node), mUnitEventSet(node),
      mLogger(node, "pe", "Processing Element Log"), mComputeTime(params->compute_time),
      mActWidth(params->act_width), mWeightWidth(params->weight_width),
      mTotalMacs(getStatisticSet(), "total_macs", "Count of MAC operations",
                 sparta::Counter::COUNT_NORMAL),
      mTickEvent(&mUnitEventSet, "tick_event", CREATE_SPARTA_HANDLER(PE, Tick)) {
    // Register port handlers
    mPortSet.inWeight.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(PE, HandleWeight, int16_t));
    mPortSet.inAct.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(PE, HandleActivation, int16_t));
    mPortSet.inWtPs.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(PE, HandleWeightPartialSum, int32_t));
    mPortSet.inWtValid.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(PE, HandleWeightValid, uint32_t));

    // Create and register tick event
    sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(PE, Tick));
}

// Direct methods to set values
void PE::SetWeight(int16_t weight) { HandleWeight(weight); }

void PE::ReceiveActivation(int16_t act) { HandleActivation(act); }

void PE::ReceiveWeightPartialSum(int32_t wtPs) { HandleWeightPartialSum(wtPs); }

void PE::SetWeightValidSignal(uint32_t valid) { HandleWeightValid(valid); }

// Handle direct weight initialization (used for testing/initialization)
void PE::HandleWeight(const int16_t & weight) {
#ifdef DEBUG_PE
    std::cout << "PE: Direct weight set: " << weight << std::endl;
#endif
    mWeightReg = weight;
}

// Handle activation input from west
void PE::HandleActivation(const int16_t & act) {
    // Always register the activation input, as per RTL
    mActReg = act;
    
    // Forward activation to east (next PE)
    mPortSet.outAct.send(mActReg);

#ifdef DEBUG_PE
    std::cout << "PE: Received activation: " << act << ", registered: " << mActReg << std::endl;
#endif

    // The actual MAC computation is triggered by ProcessData, not directly here
}

// Handle weight/partial sum from north
void PE::HandleWeightPartialSum(const int32_t & wtPs) {
    if (mWeightLoadingMode) {
        // Weight loading mode - pass weight down the column
        // During weight loading, the inWtPs contains the weight value itself
        mWeightReg = wtPs & ((1 << mWeightWidth) - 1); // Extract lower bits as weight
        mPortSet.outWtPs.send(wtPs); // Forward weight to south (next PE in column)
        
#ifdef DEBUG_PE
        std::cout << "PE: Weight loading mode - received weight: " << wtPs 
                  << ", stored: " << mWeightReg << std::endl;
#endif
    } else {
        // Computation mode - compute MAC and forward partial sum
        // Perform MAC operation (multiply-accumulate)
        int32_t mac = static_cast<int32_t>(mWeightReg) * static_cast<int32_t>(mActReg);
        mPartialSumReg = wtPs + mac;
        
        // Send partial sum to south (next PE in column)
        mPortSet.outWtPs.send(mPartialSumReg);
        
        // Count operation for statistics
        mTotalMacs++;
        
#ifdef DEBUG_PE
        std::cout << "PE: Computation mode - act: " << mActReg << ", weight: " << mWeightReg 
                  << ", mac: " << mac << ", in_ps: " << wtPs << ", out_ps: " << mPartialSumReg << std::endl;
#endif
    }
}

// Handle weight valid signal
void PE::HandleWeightValid(const uint32_t & valid) {
    // Set weight loading mode based on valid signal (1 = weight loading mode)
    mWeightLoadingMode = (valid == 1);
    
#ifdef DEBUG_PE
    std::cout << "PE: Weight valid signal: " << valid << ", weight loading mode: " 
              << (mWeightLoadingMode ? "ON" : "OFF") << std::endl;
#endif
}

// Process data - this is a placeholder for any additional processing logic
void PE::ProcessData() {
    // Start processing - set busy flag and cycle counter
    mBusy = true;
    mCycleCounter = mComputeTime;
    
#ifdef DEBUG_PE
    std::cout << "PE: Processing started, compute time: " << mComputeTime << std::endl;
#endif
}

// Tick method - process one cycle (called every clock cycle)
void PE::Tick() {
    if (mBusy) {
        if (mCycleCounter > 0) {
            --mCycleCounter;
#ifdef DEBUG_PE
            std::cout << "PE: Cycle counter: " << mCycleCounter << std::endl;
#endif
        }

        if (mCycleCounter == 0) {
            mBusy = false;
#ifdef DEBUG_PE
            std::cout << "PE: Processing complete" << std::endl;
#endif
        }
    }

    // Schedule next tick using UniqueEvent
    mTickEvent.schedule(1);
}

} // namespace gemmini