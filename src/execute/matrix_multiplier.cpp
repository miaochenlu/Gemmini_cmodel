// matrix_multiplier.cpp - Implementation of Matrix Multiplier for Gemmini using SPARTA
#include "execute/matrix_multiplier.hpp"
#include "sparta/kernel/Scheduler.hpp"
#include "sparta/kernel/SpartaHandler.hpp"
#include <algorithm>
#include <iostream>

namespace gemmini {
// Initialize static name
const char MatrixMultiplier::name[] = "matrix_multiplier";

// MatrixMultiplier Constructor
MatrixMultiplier::MatrixMultiplier(sparta::TreeNode* node,
                                   const MatrixMultiplierParameterSet* params)
    : sparta::Unit(node), mPortSet(node), mToSystolicWeights(node, "to_systolic_weights"),
      mToSystolicVector(node, "to_systolic_vector"),
      mFromSystolicResults(node, "from_systolic_results", sparta::SchedulingPhase::Tick, 0),
      mUnitEventSet(node), mLogger(node, "matrix_multiplier", "Matrix Multiplier Log"),
      mSystolicRows(params->systolic_rows), mSystolicCols(params->systolic_cols),
      mTotalMms(getStatisticSet(), "total_mms", "Count of matrix multiplications",
                sparta::Counter::COUNT_NORMAL),
      mTotalBlocks(getStatisticSet(), "total_blocks", "Count of block operations",
                   sparta::Counter::COUNT_NORMAL) {
    // Register port handlers
    mPortSet.in_matrix_a.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(MatrixMultiplier, HandleMatrixA, MatrixPtr));
    mPortSet.in_matrix_b.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(MatrixMultiplier, HandleMatrixB, MatrixPtr));
    mPortSet.in_control.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(MatrixMultiplier, HandleControl, uint32_t));
    mFromSystolicResults.registerConsumerHandler(
        CREATE_SPARTA_HANDLER_WITH_DATA(MatrixMultiplier, HandleSystolicResults, MatrixPtr));

    // Create systolic array child
    sparta::TreeNode* systolicNode = new sparta::TreeNode(node, "systolic_array", "Systolic Array");

    // Create systolic array parameter set
    auto paramsForSystolic = new SystolicArrayParameterSet(systolicNode);

    // No need to manually set parameters - they have default values

    // Create systolic array resource
    SystolicArray::Factory systolicFactory;
    SystolicArray* systolicArray = static_cast<SystolicArray*>(
        systolicFactory.createResource(systolicNode, paramsForSystolic));

    // Connect ports
    mToSystolicWeights.bind(systolicArray->GetPortSet().in_weights);
    mToSystolicVector.bind(systolicArray->GetPortSet().in_vector);
    mFromSystolicResults.bind(systolicArray->GetPortSet().out_results);
}

// Handle receiving matrix A
void MatrixMultiplier::HandleMatrixA(const MatrixPtr & a) {
// Avoid using logger_.debug() or similar
#ifdef DEBUG_MATRIX_MULTIPLIER
    std::cout << "Received matrix A: " << a->Rows() << "x" << a->Cols() << std::endl;
#endif
    mMatrixA = a;
}

// Handle receiving matrix B
void MatrixMultiplier::HandleMatrixB(const MatrixPtr & b) {
#ifdef DEBUG_MATRIX_MULTIPLIER
    std::cout << "Received matrix B: " << b->Rows() << "x" << b->Cols() << std::endl;
#endif
    mMatrixB = b;
}

// Handle control signals
void MatrixMultiplier::HandleControl(const uint32_t & signal) {
#ifdef DEBUG_MATRIX_MULTIPLIER
    std::cout << "Received control signal: " << signal << std::endl;
#endif

    // If it's a start signal, start matrix multiplication
    if (signal == 1 && mMatrixA && mMatrixB) {
        StartMultiplication();
    }
}

// Multiply two matrices - direct method for simulation
void MatrixMultiplier::Multiply(const MatrixPtr & a, const MatrixPtr & b) {
    // Pass matrices to the port handlers
    HandleMatrixA(a);
    HandleMatrixB(b);

    // Start multiplication process
    StartMultiplication();
}

// Start the matrix multiplication process
void MatrixMultiplier::StartMultiplication() {
    if (mBusy) {
#ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Matrix multiplier is busy, ignoring new multiplication request" << std::endl;
#endif
        return;
    }

    // Check compatibility of matrices
    if (mMatrixA->Cols() != mMatrixB->Rows()) {
        std::cerr << "Matrix dimensions incompatible for multiplication: " << mMatrixA->Rows()
                  << "x" << mMatrixA->Cols() << " * " << mMatrixB->Rows() << "x" << mMatrixB->Cols()
                  << std::endl;
        return;
    }

#ifdef DEBUG_MATRIX_MULTIPLIER
    std::cout << "Starting matrix multiplication: " << mMatrixA->Rows() << "x" << mMatrixA->Cols()
              << " * " << mMatrixB->Rows() << "x" << mMatrixB->Cols() << std::endl;
#endif

    // Set busy flag
    mBusy = true;

    // Reset block counters
    mCurrentRowBlock = 0;
    mCurrentColBlock = 0;

    // Calculate number of blocks needed
    mTotalRowBlocks = (mMatrixA->Rows() + mSystolicRows - 1) / mSystolicRows;
    mTotalColBlocks = (mMatrixB->Cols() + mSystolicCols - 1) / mSystolicCols;

    // Initialize result matrix
    mResultMatrix = CreateMatrixPtr<Matrix>(mMatrixA->Rows(), mMatrixB->Cols());

    // Start processing the first block
    ProcessNextBlock();

    // Update statistics
    mTotalMms++;
}

// Process next block in the matrix multiplication
void MatrixMultiplier::ProcessNextBlock() {
    // Calculate block dimensions and offsets
    uint32_t rowOffset = mCurrentRowBlock * mSystolicRows;
    uint32_t colOffset = mCurrentColBlock * mSystolicCols;

    uint32_t blockRows = std::min(mSystolicRows, mMatrixA->Rows() - rowOffset);
    uint32_t blockCols = std::min(mSystolicCols, mMatrixB->Cols() - colOffset);

#ifdef DEBUG_MATRIX_MULTIPLIER
    std::cout << "Processing block [" << mCurrentRowBlock << "," << mCurrentColBlock
              << "]: " << blockRows << "x" << blockCols << std::endl;
#endif

    // Create weight matrix for this block (transposed portion of B)
    MatrixPtr weights = CreateMatrixPtr<Matrix>(mSystolicRows, mSystolicCols);

    // Fill weight matrix with transposed values from B
    for (uint32_t r = 0; r < blockRows; ++r) {
        for (uint32_t c = 0; c < blockCols; ++c) {
            if (rowOffset + r < mMatrixA->Rows() && c < mMatrixB->Rows()) {
                // Transpose during load
                weights->At(r, c) = mMatrixB->At(c, colOffset + r);
            }
        }
    }

    // Send weights to systolic array
    mToSystolicWeights.send(weights);

    // Create input vectors for this block (from A)
    std::vector<VectorPtr> inputVectors;

    // For each row in the block from A
    for (uint32_t r = 0; r < blockRows; ++r) {
        VectorPtr rowVector = CreateMatrixPtr<Vector>(mMatrixA->Cols());

        // Fill vector with values from A
        for (uint32_t c = 0; c < mMatrixA->Cols(); ++c) {
            if (rowOffset + r < mMatrixA->Rows()) {
                (*rowVector)[c] = mMatrixA->At(rowOffset + r, c);
            }
        }

        inputVectors.push_back(rowVector);
    }

    // Process each input vector
    for (const auto & vector : inputVectors) {
        // Send vector to systolic array
        mToSystolicVector.send(vector);
    }

    // Update statistics
    mTotalBlocks++;
}

// Handle results from systolic array
void MatrixMultiplier::HandleSystolicResults(const MatrixPtr & results) {
    if (!mBusy) {
#ifdef DEBUG_MATRIX_MULTIPLIER
        std::cout << "Received results when not busy, ignoring" << std::endl;
#endif
        return;
    }

#ifdef DEBUG_MATRIX_MULTIPLIER
    std::cout << "Received results for block [" << mCurrentRowBlock << "," << mCurrentColBlock
              << "]" << std::endl;
#endif

    // Copy block results to the appropriate position in the final result matrix
    uint32_t rowOffset = mCurrentRowBlock * mSystolicRows;
    uint32_t colOffset = mCurrentColBlock * mSystolicCols;

    uint32_t blockRows = std::min(mSystolicRows, mMatrixA->Rows() - rowOffset);
    uint32_t blockCols = std::min(mSystolicCols, mMatrixB->Cols() - colOffset);

    // Copy results
    for (uint32_t r = 0; r < blockRows; ++r) {
        for (uint32_t c = 0; c < blockCols; ++c) {
            mResultMatrix->At(rowOffset + r, colOffset + c) = results->At(r, c);
        }
    }

    // Move to next block
    mCurrentColBlock++;
    if (mCurrentColBlock >= mTotalColBlocks) {
        mCurrentColBlock = 0;
        mCurrentRowBlock++;

        if (mCurrentRowBlock >= mTotalRowBlocks) {
            // All blocks processed
            MultiplierDone();
            return;
        }
    }

    // Process next block
    ProcessNextBlock();
}

// Called when all blocks have been processed
void MatrixMultiplier::MultiplierDone() {
#ifdef DEBUG_MATRIX_MULTIPLIER
    std::cout << "Matrix multiplication complete" << std::endl;
#endif

    // Send result to output port
    mPortSet.out_result.send(mResultMatrix);

    // Reset busy flag
    mBusy = false;
}

} // namespace gemmini