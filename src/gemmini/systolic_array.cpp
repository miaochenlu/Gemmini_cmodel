// systolic_array.cpp - Implementation of Systolic Array for Gemmini simulator using SPARTA
#include "gemmini/systolic_array.hpp"
#include "sparta/events/StartupEvent.hpp"
#include "sparta/kernel/Scheduler.hpp"
#include "sparta/kernel/SpartaHandler.hpp"
#include <iostream>

namespace gemmini {
// Initialize static name
const char SystolicArray::name[] = "systolic_array";

// SystolicArray Constructor
SystolicArray::SystolicArray(sparta::TreeNode* node, const SystolicArrayParameterSet* params)
    : sparta::Unit(node), mPortSet(node), mUnitEventSet(node),
      mLogger(node, "systolic_array", "Systolic Array Log"), mRows(params->rows),
      mCols(params->cols), mComputeTime(params->compute_time),
      mTotalMatrixOps(getStatisticSet(), "total_matrix_ops", "Count of matrix operations",
                      sparta::Counter::COUNT_NORMAL),
      mTickEvent(&mUnitEventSet, "tick_event", CREATE_SPARTA_HANDLER(SystolicArray, Tick)) {
    // Register port handlers
    mPortSet.in_weights.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, HandleWeights, MatrixPtr));
    mPortSet.in_vector.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, HandleVector, VectorPtr));
    mPortSet.in_control.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, HandleControl, uint32_t));

    // Create 4x4 grid of Processing Elements
    std::cout << "SystolicArray: Creating " << mRows << "x" << mCols << " systolic array"
              << std::endl;

    for (uint32_t r = 0; r < mRows; ++r) {
        for (uint32_t c = 0; c < mCols; ++c) {
            // Create PE node with a unique name
            std::string peName = GetPEName(r, c);
            std::string peDesc =
                "Processing Element at (" + std::to_string(r) + "," + std::to_string(c) + ")";
            sparta::TreeNode* peNode = new sparta::TreeNode(node, peName, peDesc);

            // Create PE parameter set
            auto peParams = new PEParameterSet(peNode);
            // Set PE compute time to match systolic array
            peParams->compute_time = mComputeTime;
            peParams->act_width = 16;  // Default activation width
            peParams->weight_width = 16; // Default weight width

            // Create PE using factory
            PE::Factory peFactory;
            PE* pe = static_cast<PE*>(peFactory.createResource(peNode, peParams));

            // Store PE pointer
            mPEs.push_back(pe);

#ifdef DEBUG_SYSTOLIC_ARRAY
            std::cout << "SystolicArray: Created PE at (" << r << "," << c << ")" << std::endl;
#endif
        }
    }

    // Connect PEs in the systolic array (data flows horizontally, results accumulate vertically)
    std::cout << "SystolicArray: Connecting PEs in the systolic array" << std::endl;

    for (uint32_t r = 0; r < mRows; ++r) {
        for (uint32_t c = 0; c < mCols; ++c) {
            // Get current PE
            PE* currentPe = GetPE(r, c);

            // Connect to PE on the right (data forwarding, if not last column)
            if (c < mCols - 1) {
                PE* rightPe = GetPE(r, c + 1);
                currentPe->GetPortSet().outAct.bind(rightPe->GetPortSet().inAct);
#ifdef DEBUG_SYSTOLIC_ARRAY
                std::cout << "SystolicArray: Connected PE(" << r << "," << c << ") -> PE(" << r
                          << "," << (c + 1) << ")" << std::endl;
#endif
            }

            // Connect to PE below (partial sum forwarding, if not last row)
            if (r < mRows - 1) {
                PE* belowPe = GetPE(r + 1, c);
                currentPe->GetPortSet().outWtPs.bind(belowPe->GetPortSet().inWtPs);
#ifdef DEBUG_SYSTOLIC_ARRAY
                std::cout << "SystolicArray: Connected PE(" << r << "," << c << ") -> PE(" << (r + 1)
                          << "," << c << ") for partial sums" << std::endl;
#endif
            }
        }
    }

    // Initialize result matrix
    mResultMatrix = CreateMatrixPtr<Matrix>(mRows, 1); // Result is a column vector

    // Create and register tick event
    sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(SystolicArray, Tick));

    std::cout << "SystolicArray: Systolic Array initialization complete" << std::endl;
}

// Handle weights matrix
void SystolicArray::HandleWeights(const MatrixPtr & weights) {
    // For a 4x4 systolic array, the weights should be 4x4
    if (weights->Rows() != mRows || weights->Cols() != mCols) {
        std::cerr << "SystolicArray: Weight matrix dimensions (" << weights->Rows() << "x"
                  << weights->Cols() << ") don't match systolic array dimensions (" << mRows << "x"
                  << mCols << ")" << std::endl;
        return;
    }

    std::cout << "SystolicArray: Loading weights into PEs" << std::endl;

    // First, set weight loading mode for all PEs
    for (uint32_t r = 0; r < mRows; ++r) {
        for (uint32_t c = 0; c < mCols; ++c) {
            PE* pe = GetPE(r, c);
            pe->SetWeightValidSignal(1); // Enter weight loading mode
        }
    }

    // Load weights into each PE (stationary weights in weight-stationary design)
    for (uint32_t r = 0; r < mRows; ++r) {
        for (uint32_t c = 0; c < mCols; ++c) {
            // Get the PE at this position
            PE* pe = GetPE(r, c);

            // Send weight directly to the PE
            pe->SetWeight(weights->At(r, c));

#ifdef DEBUG_SYSTOLIC_ARRAY
            std::cout << "SystolicArray: PE(" << r << "," << c << ") weight set to "
                      << weights->At(r, c) << std::endl;
#endif
        }
    }

    // Exit weight loading mode
    for (uint32_t r = 0; r < mRows; ++r) {
        for (uint32_t c = 0; c < mCols; ++c) {
            PE* pe = GetPE(r, c);
            pe->SetWeightValidSignal(0); // Exit weight loading mode
        }
    }
}

// Handle input vector
void SystolicArray::HandleVector(const VectorPtr & input) {
    if (mProcessing) {
        std::cerr << "SystolicArray: Still processing, ignoring new input vector" << std::endl;
        return;
    }

    // For a 4x4 systolic array, the input vector should be size 4
    if (input->Size() != mRows) {
        std::cerr << "SystolicArray: Input vector size (" << input->Size()
                  << ") doesn't match systolic array rows (" << mRows << ")" << std::endl;
        return;
    }

    std::cout << "SystolicArray: Starting processing of input vector" << std::endl;

    // Reset result matrix
    mResultMatrix = CreateMatrixPtr<Matrix>(mRows, 1);

    // Reset each PE's accumulator (send reset control signal)
    for (uint32_t r = 0; r < mRows; ++r) {
        for (uint32_t c = 0; c < mCols; ++c) {
            GetPE(r, c)->SetWeightValidSignal(0); // Make sure we're in computation mode
        }
    }

    // Save input for processing
    mCurrentInput = input;

    // Start processing
    mProcessing = true;
    mCurrentCycle = 0;

    // Calculate total cycles needed (for skewed scheduling)
    // For an NxN array with 1-cycle compute time, we need 2N-1 cycles for input flow plus
    // compute_time_
    mTotalCyclesNeeded = mRows + mCols - 1 + mComputeTime;

    std::cout << "SystolicArray: Processing will take " << mTotalCyclesNeeded << " cycles"
              << std::endl;
}

// Handle control signals
void SystolicArray::HandleControl(const uint32_t & signal) {
#ifdef DEBUG_SYSTOLIC_ARRAY
    std::cout << "SystolicArray: Received control signal: " << signal << std::endl;
#endif

    // If it's a completion check signal
    if (signal == 1) {
        if (mProcessing && mCurrentCycle >= mTotalCyclesNeeded) {
            ComputationComplete();
        } else if (mProcessing) {
            std::cout << "SystolicArray: Still processing: " << mCurrentCycle << "/"
                      << mTotalCyclesNeeded << " cycles" << std::endl;
        } else {
            std::cout << "SystolicArray: Not currently processing" << std::endl;
        }
    }
}

// Process one cycle of the systolic array
void SystolicArray::ProcessOneCycle() {
    if (!mProcessing) {
        return;
    }

#ifdef DEBUG_SYSTOLIC_ARRAY
    std::cout << "SystolicArray: Processing cycle " << mCurrentCycle << " of " << mTotalCyclesNeeded
              << std::endl;
#endif

    // For each cycle, determine which PEs should receive inputs
    // This implements the diagonal wavefront pattern of systolic arrays

    // For each row
    for (uint32_t r = 0; r < mRows; ++r) {
        // Calculate which column should receive data this cycle (based on skewed scheduling)
        // The formula accounts for the diagonal wavefront pattern
        int32_t activeCol = mCurrentCycle - r;

        if (activeCol >= 0 && activeCol < static_cast<int32_t>(mCols)) {
            // Get the PE that should receive data this cycle
            PE* pe = GetPE(r, activeCol);

            // Calculate index into input vector
            uint32_t inputIdx = r;

            // Ensure we don't access beyond input vector size
            if (inputIdx < mCurrentInput->Size()) {
                // Send data to the PE
                pe->ReceiveActivation((*mCurrentInput)[inputIdx]);
#ifdef DEBUG_SYSTOLIC_ARRAY
                std::cout << "SystolicArray: Sent input " << (*mCurrentInput)[inputIdx] << " to PE("
                          << r << "," << activeCol << ")" << std::endl;
#endif
            }
        }
    }

    // Increment cycle counter
    mCurrentCycle++;

    // Check if processing is complete
    if (mCurrentCycle >= mTotalCyclesNeeded) {
        ComputationComplete();
    }
}

// Called when computation is complete
void SystolicArray::ComputationComplete() {
    std::cout << "SystolicArray: Computation complete, collecting results" << std::endl;

    // For a weight-stationary design, the results are in the rightmost column
    // Extract results from rightmost PEs
    for (uint32_t r = 0; r < mRows; ++r) {
        // Get PE at rightmost column
        PE* pe = GetPE(r, mCols - 1);

        // In a real implementation, we would need to handle the response from the PE asynchronously
        // This is simplified for clarity
        // The PE would send its accumulated value to out_result port, which we'd capture

        // For now, just simulate setting the result
        mResultMatrix->At(r, 0) = 0; // This would be replaced with actual result from PE
    }

    // Send results matrix
    mPortSet.out_results.send(mResultMatrix);
    std::cout << "SystolicArray: Sent result matrix" << std::endl;

    // Update statistics
    mTotalMatrixOps++;

    // Reset processing flag
    mProcessing = false;
}

// Tick method - called every clock cycle
void SystolicArray::Tick() {
    // Process one cycle of the systolic array
    ProcessOneCycle();

    // Schedule next tick using UniqueEvent
    mTickEvent.schedule(1);
}

} // namespace gemmini