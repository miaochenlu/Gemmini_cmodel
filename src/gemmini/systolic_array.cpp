// systolic_array.cpp - Implementation of Systolic Array for Gemmini simulator using SPARTA
#include "gemmini/systolic_array.hpp"
#include "sparta/events/StartupEvent.hpp"
#include "sparta/kernel/Scheduler.hpp"
#include "sparta/kernel/SpartaHandler.hpp"
#include <iostream>

namespace gemmini
{
    // Initialize static name
    const char SystolicArray::name[] = "systolic_array";
    
    // SystolicArray Constructor
    SystolicArray::SystolicArray(sparta::TreeNode* node, const SystolicArrayParameterSet* params) :
        sparta::Unit(node),
        port_set_(node),
        unit_event_set_(node),
        logger_(node, "systolic_array", "Systolic Array Log"),
        rows_(params->rows),
        cols_(params->cols),
        compute_time_(params->compute_time),
        total_matrix_ops_(getStatisticSet(), "total_matrix_ops", "Count of matrix operations", sparta::Counter::COUNT_NORMAL),
        tick_event_(&unit_event_set_, "tick_event", CREATE_SPARTA_HANDLER(SystolicArray, tick_))
    {
        // Register port handlers
        port_set_.in_weights.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, handleWeights_, MatrixPtr));
        port_set_.in_vector.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, handleVector_, VectorPtr));
        port_set_.in_control.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, handleControl_, uint32_t));
        
        // Create 4x4 grid of Processing Elements
        std::cout << "SystolicArray: Creating " << rows_ << "x" << cols_ << " systolic array" << std::endl;
        
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                // Create PE node with a unique name
                std::string pe_name = getPEName(r, c);
                std::string pe_desc = "Processing Element at (" + std::to_string(r) + "," + std::to_string(c) + ")";
                sparta::TreeNode* pe_node = new sparta::TreeNode(node, pe_name, pe_desc);
                
                // Create PE parameter set
                auto pe_params = new PEParameterSet(pe_node);
                
                // Set PE compute time to match systolic array
                pe_params->compute_time = compute_time_;
                
                // Create PE using factory
                PE::Factory pe_factory;
                PE* pe = static_cast<PE*>(pe_factory.createResource(pe_node, pe_params));
                
                // Store PE pointer
                pes_.push_back(pe);
                
                #ifdef DEBUG_SYSTOLIC_ARRAY
                std::cout << "SystolicArray: Created PE at (" << r << "," << c << ")" << std::endl;
                #endif
            }
        }
        
        // Connect PEs in the systolic array (data flows horizontally, results accumulate vertically)
        std::cout << "SystolicArray: Connecting PEs in the systolic array" << std::endl;
        
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                // Get current PE
                PE* current_pe = getPE(r, c);
                
                // Connect to PE on the right (data forwarding, if not last column)
                if (c < cols_ - 1) {
                    PE* right_pe = getPE(r, c + 1);
                    current_pe->getPortSet().out_data.bind(right_pe->getPortSet().in_data);
                    #ifdef DEBUG_SYSTOLIC_ARRAY
                    std::cout << "SystolicArray: Connected PE(" << r << "," << c << ") -> PE(" << r << "," << (c+1) << ")" << std::endl;
                    #endif
                }
            }
        }
        
        // Initialize result matrix
        result_matrix_ = create_matrix_ptr<Matrix>(rows_, 1);  // Result is a column vector
        
        // Create and register tick event
        sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(SystolicArray, tick_));
        
        std::cout << "SystolicArray: Systolic Array initialization complete" << std::endl;
    }

    // Handle weights matrix
    void SystolicArray::handleWeights_(const MatrixPtr& weights)
    {
        // For a 4x4 systolic array, the weights should be 4x4
        if (weights->rows() != rows_ || weights->cols() != cols_) {
            std::cerr << "SystolicArray: Weight matrix dimensions (" << weights->rows() << "x" << weights->cols() 
                    << ") don't match systolic array dimensions (" << rows_ << "x" << cols_ << ")" << std::endl;
            return;
        }
        
        std::cout << "SystolicArray: Loading weights into PEs" << std::endl;
        
        // Load weights into each PE (stationary weights in weight-stationary design)
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                // Get the PE at this position
                PE* pe = getPE(r, c);
                
                // Send weight directly to the PE
                pe->setWeight(weights->at(r, c));
                
                #ifdef DEBUG_SYSTOLIC_ARRAY
                std::cout << "SystolicArray: PE(" << r << "," << c << ") weight set to " << weights->at(r, c) << std::endl;
                #endif
            }
        }
    }

    // Handle input vector
    void SystolicArray::handleVector_(const VectorPtr& input)
    {
        if (processing_) {
            std::cerr << "SystolicArray: Still processing, ignoring new input vector" << std::endl;
            return;
        }
        
        // For a 4x4 systolic array, the input vector should be size 4
        if (input->size() != rows_) {
            std::cerr << "SystolicArray: Input vector size (" << input->size() 
                    << ") doesn't match systolic array rows (" << rows_ << ")" << std::endl;
            return;
        }
        
        std::cout << "SystolicArray: Starting processing of input vector" << std::endl;
        
        // Reset result matrix
        result_matrix_ = create_matrix_ptr<Matrix>(rows_, 1);
        
        // Reset each PE's accumulator (send reset control signal)
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                getPE(r, c)->controlSignal(1); // Reset signal
            }
        }
        
        // Save input for processing
        current_input_ = input;
        
        // Start processing
        processing_ = true;
        current_cycle_ = 0;
        
        // Calculate total cycles needed (for skewed scheduling)
        // For an NxN array with 1-cycle compute time, we need 2N-1 cycles for input flow plus compute_time_
        total_cycles_needed_ = rows_ + cols_ - 1 + compute_time_;
        
        std::cout << "SystolicArray: Processing will take " << total_cycles_needed_ << " cycles" << std::endl;
    }

    // Handle control signals
    void SystolicArray::handleControl_(const uint32_t& signal)
    {
        #ifdef DEBUG_SYSTOLIC_ARRAY
        std::cout << "SystolicArray: Received control signal: " << signal << std::endl;
        #endif
        
        // If it's a completion check signal
        if (signal == 1) {
            if (processing_ && current_cycle_ >= total_cycles_needed_) {
                computationComplete_();
            } else if (processing_) {
                std::cout << "SystolicArray: Still processing: " << current_cycle_ << "/" << total_cycles_needed_ << " cycles" << std::endl;
            } else {
                std::cout << "SystolicArray: Not currently processing" << std::endl;
            }
        }
    }

    // Process one cycle of the systolic array
    void SystolicArray::processOneCycle_()
    {
        if (!processing_) {
            return;
        }
        
        #ifdef DEBUG_SYSTOLIC_ARRAY
        std::cout << "SystolicArray: Processing cycle " << current_cycle_ << " of " << total_cycles_needed_ << std::endl;
        #endif
        
        // For each cycle, determine which PEs should receive inputs
        // This implements the diagonal wavefront pattern of systolic arrays
        
        // For each row
        for (uint32_t r = 0; r < rows_; ++r) {
            // Calculate which column should receive data this cycle (based on skewed scheduling)
            // The formula accounts for the diagonal wavefront pattern
            int32_t active_col = current_cycle_ - r;
            
            if (active_col >= 0 && active_col < static_cast<int32_t>(cols_)) {
                // Get the PE that should receive data this cycle
                PE* pe = getPE(r, active_col);
                
                // Calculate index into input vector
                uint32_t input_idx = r;
                
                // Ensure we don't access beyond input vector size
                if (input_idx < current_input_->size()) {
                    // Send data to the PE
                    pe->receiveData((*current_input_)[input_idx]);
                    #ifdef DEBUG_SYSTOLIC_ARRAY
                    std::cout << "SystolicArray: Sent input " << (*current_input_)[input_idx] << " to PE(" << r << "," << active_col << ")" << std::endl;
                    #endif
                }
            }
        }
        
        // Increment cycle counter
        current_cycle_++;
        
        // Check if processing is complete
        if (current_cycle_ >= total_cycles_needed_) {
            computationComplete_();
        }
    }

    // Called when computation is complete
    void SystolicArray::computationComplete_()
    {
        std::cout << "SystolicArray: Computation complete, collecting results" << std::endl;
        
        // For a weight-stationary design, the results are in the rightmost column
        // Extract results from rightmost PEs
        for (uint32_t r = 0; r < rows_; ++r) {
            // Get PE at rightmost column
            PE* pe = getPE(r, cols_ - 1);
            
            // Send read signal to PE to get the result
            pe->controlSignal(2);
            
            // In a real implementation, we would need to handle the response from the PE asynchronously
            // This is simplified for clarity
            // The PE would send its accumulated value to out_result port, which we'd capture
            
            // For now, just simulate setting the result
            result_matrix_->at(r, 0) = 0; // This would be replaced with actual result from PE
        }
        
        // Send results matrix
        port_set_.out_results.send(result_matrix_);
        std::cout << "SystolicArray: Sent result matrix" << std::endl;
        
        // Update statistics
        total_matrix_ops_++;
        
        // Reset processing flag
        processing_ = false;
    }

    // Tick method - called every clock cycle
    void SystolicArray::tick_()
    {
        // Process one cycle of the systolic array
        processOneCycle_();
        
        // Schedule next tick using UniqueEvent
        tick_event_.schedule(1);
    }
    
} // namespace gemmini 