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
      mLogger(node, "pe", "Processing Element Log"), 
      mComputeCycles(params->compute_cycles),
      mActWidth(params->act_width), 
      mWeightWidth(params->weight_width),
      mDelayCycles(params->delay_cycles),
      mDebugFifo(params->debug_fifo),
      mTotalMacs(getStatisticSet(), "total_macs", "Count of MAC operations",
                 sparta::Counter::COUNT_NORMAL),
      mTickEvent(&mUnitEventSet, "tick_event", CREATE_SPARTA_HANDLER(PE, Tick)) {
    // Initialize output state
    mOutput.act = 0;
    mOutput.psum = 0;
    mOutput.act_valid = false;
    mOutput.psum_valid = false;
    
    // Register port handlers
    mPortSet.inputs.weight.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(PE, HandleWeight, int16_t));
    mPortSet.inputs.act.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(PE, HandleActivation, int16_t));
    mPortSet.inputs.partialSum.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(PE, HandlePartialSum, int32_t));

    // Create delay FIFOs for activation and partial sum
    // Create activation delay FIFO
    std::string act_fifo_name = "act_delay_fifo";
    sparta::TreeNode* act_fifo_node = new sparta::TreeNode(node, act_fifo_name, "Activation Delay FIFO");
    DelayFifoParameterSet<int16_t>* act_fifo_params = new DelayFifoParameterSet<int16_t>(act_fifo_node);
    act_fifo_params->depth = mDelayCycles;
    act_fifo_params->debug_mode = mDebugFifo;
    
    // Create and initialize the activation delay FIFO
    DelayFifo<int16_t>::Factory act_fifo_factory;
    DelayFifo<int16_t>* act_fifo = static_cast<DelayFifo<int16_t>*>(
        act_fifo_factory.createResource(act_fifo_node, act_fifo_params));
    mActDelayFifo.reset(act_fifo);
    
    // Connect FIFO output to PE activation output port
    mActDelayFifo->GetPortSet().out.bind(&mPortSet.outputs.act);
    
    // Create partial sum delay FIFO
    std::string psum_fifo_name = "psum_delay_fifo";
    sparta::TreeNode* psum_fifo_node = new sparta::TreeNode(node, psum_fifo_name, "Partial Sum Delay FIFO");
    DelayFifoParameterSet<int32_t>* psum_fifo_params = new DelayFifoParameterSet<int32_t>(psum_fifo_node);
    psum_fifo_params->depth = mDelayCycles;
    psum_fifo_params->debug_mode = mDebugFifo;
    
    // Create and initialize the partial sum delay FIFO
    DelayFifo<int32_t>::Factory psum_fifo_factory;
    DelayFifo<int32_t>* psum_fifo = static_cast<DelayFifo<int32_t>*>(
        psum_fifo_factory.createResource(psum_fifo_node, psum_fifo_params));
    mPsumDelayFifo.reset(psum_fifo);
    
    // Connect FIFO output to PE partial sum output port
    mPsumDelayFifo->GetPortSet().out.bind(&mPortSet.outputs.partialSum);

    // Create and register tick event
    sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(PE, Tick));
    
    if (mDebugFifo) {
        std::cout << "PE created with " << mDelayCycles << " cycle(s) of delay between PEs" << std::endl;
    }
}

// Direct methods to set values
void PE::SetWeight(int16_t weight) { HandleWeight(weight); }

void PE::ReceiveActivation(int16_t act) { HandleActivation(act); }

void PE::ReceivePartialSum(int32_t partialSum) { HandlePartialSum(partialSum); }

// Handle weight preloading (used for initialization)
void PE::HandleWeight(const int16_t & weight) {
#ifdef DEBUG_PE
    std::cout << "PE: Weight set: " << weight << std::endl;
#endif
    mWeightReg = weight;
}

// Handle activation input from west
void PE::HandleActivation(const int16_t & act) {
    // Store activation input
    mInput.act = act;
    mInput.act_valid = true;
    
    // Set activation output
    mOutput.act = act;
    mOutput.act_valid = true;
    
    // Push activation to the delay FIFO for propagation to the next PE
    mActDelayFifo->Push(mOutput.act);

#ifdef DEBUG_PE
    std::cout << "PE: Received activation: " << act << std::endl;
#endif

    // Check if all inputs are ready for computation
    if (CanCompute()) {
        ComputeMAC();
    }
}

// Handle partial sum from north
void PE::HandlePartialSum(const int32_t & partialSum) {
    // Store partial sum input
    mInput.psum = partialSum;
    mInput.psum_valid = true;
    
    // Initialize output partial sum (will be updated in ComputeMAC)
    mOutput.psum = partialSum;
    mOutput.psum_valid = false;
    
#ifdef DEBUG_PE
    std::cout << "PE: Received partial sum: " << partialSum << std::endl;
#endif

    // Check if all inputs are ready for computation
    if (CanCompute()) {
        ComputeMAC();
    }
}

// Compute MAC - multiply activation with weight and accumulate with partial sum
void PE::ComputeMAC() {
    // Perform MAC operation (multiply-accumulate)
    int32_t product = static_cast<int32_t>(mWeightReg) * static_cast<int32_t>(mInput.act);
    
    // Initialize result with incoming partial sum
    mPartialSumReg = mInput.psum;
    
    // Add product to the partial sum
    mPartialSumReg += product;
    
    // Set output values
    mOutput.psum = mPartialSumReg;
    mOutput.psum_valid = true;
    mOutput.act = mInput.act;
    mOutput.act_valid = true;
    
    // Reset input valid flags for next computation
    mInput.act_valid = false;
    mInput.psum_valid = false;
    
    // Count operation for statistics
    mTotalMacs++;
    
#ifdef DEBUG_PE
    std::cout << "PE: MAC - act: " << mInput.act << ", weight: " << mWeightReg 
              << ", incoming psum: " << mInput.psum 
              << ", product: " << product 
              << ", result: " << mPartialSumReg << std::endl;
#endif

    // If compute time > 0, set busy status for delayed computation
    if (mComputeCycles > 0) {
        mBusy = true;
        mCycleCounter = mComputeCycles;
    } else {
        // Push the result to the delay FIFO for immediate propagation
        mPsumDelayFifo->Push(mOutput.psum);
    }
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
            // Computation complete, push result to delay FIFO
            mPsumDelayFifo->Push(mOutput.psum);
            
            mBusy = false;
#ifdef DEBUG_PE
            std::cout << "PE: Processing complete, partial sum: " << mOutput.psum 
                      << " pushed to delay FIFO" << std::endl;
#endif
        }
    }

    // Schedule next tick using UniqueEvent
    mTickEvent.schedule(1);
}

} // namespace gemmini