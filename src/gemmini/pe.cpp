// pe.cpp - Implementation of Processing Element for Gemmini Systolic Array using SPARTA
#include "gemmini/pe.hpp"
#include "sparta/events/StartupEvent.hpp"
#include "sparta/kernel/Scheduler.hpp"
#include "sparta/kernel/SpartaHandler.hpp"
#include <iostream>

namespace gemmini
{
    // Initialize static name
    const char PE::name[] = "pe";
    
    // PE Constructor
    PE::PE(sparta::TreeNode* node, const PEParameterSet* params) :
        sparta::Unit(node),
        port_set_(node),
        unit_event_set_(node),
        logger_(node, "pe", "Processing Element Log"),
        compute_time_(params->compute_time),
        data_width_(params->data_width),
        total_macs_(getStatisticSet(), "total_macs", "Count of MAC operations", sparta::Counter::COUNT_NORMAL),
        tick_event_(&unit_event_set_, "tick_event", CREATE_SPARTA_HANDLER(PE, tick_))
    {
        // Register port handlers
        port_set_.in_weight.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(PE, handleWeight_, int16_t));
        port_set_.in_data.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(PE, handleData_, int16_t));
        port_set_.in_control.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(PE, handleControl_, uint32_t));
        
        // Create and register tick event
        sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(PE, tick_));
    }
    
    // Direct methods to set values
    void PE::setWeight(int16_t weight) {
        handleWeight_(weight);
    }
    
    void PE::receiveData(int16_t data) {
        handleData_(data);
    }
    
    void PE::controlSignal(uint32_t signal) {
        handleControl_(signal);
    }
    
    // Handle weight input
    void PE::handleWeight_(const int16_t& weight)
    {
        #ifdef DEBUG_PE
        std::cout << "PE: Received weight value: " << weight << std::endl;
        #endif
        weight_ = weight;
    }
    
    // Handle data input
    void PE::handleData_(const int16_t& data)
    {
        if (busy_) {
            #ifdef DEBUG_PE
            std::cout << "PE: is busy, ignoring input data: " << data << std::endl;
            #endif
            return;
        }
        
        #ifdef DEBUG_PE
        std::cout << "PE: Received input data: " << data << std::endl;
        #endif
        
        input_ = data;
        
        // Set output (for forwarding to next PE)
        output_ = input_;
        
        // Forward data to next PE
        port_set_.out_data.send(output_);
        
        // Start MAC computation
        computeResult_();
    }
    
    // Handle control signals
    void PE::handleControl_(const uint32_t& signal)
    {
        #ifdef DEBUG_PE
        std::cout << "PE: Received control signal: " << signal << std::endl;
        #endif
        
        // If it's a reset signal, reset the accumulator
        if (signal == 1) {
            accumulator_ = 0;
            #ifdef DEBUG_PE
            std::cout << "PE: Accumulator reset to 0" << std::endl;
            #endif
        }
        
        // If it's a read signal, send accumulated result
        if (signal == 2) {
            port_set_.out_result.send(accumulator_);
            #ifdef DEBUG_PE
            std::cout << "PE: Sent accumulated result: " << accumulator_ << std::endl;
            #endif
        }
    }
    
    // Compute MAC result
    void PE::computeResult_()
    {
        // Start computation process
        busy_ = true;
        cycle_counter_ = compute_time_;
        
        // Perform MAC operation (multiply-accumulate)
        int32_t product = static_cast<int32_t>(weight_) * static_cast<int32_t>(input_);
        accumulator_ += product;
        
        // Count operation for statistics
        total_macs_++;
        
        #ifdef DEBUG_PE
        std::cout << "PE: Performed MAC: " << weight_ << " * " << input_ 
                  << " = " << product 
                  << ", accumulator = " << accumulator_ << std::endl;
        #endif
    }
    
    // Tick method - process one cycle (called every clock cycle)
    void PE::tick_()
    {
        if (busy_) {
            if (cycle_counter_ > 0) {
                --cycle_counter_;
                #ifdef DEBUG_PE
                std::cout << "PE: Cycle counter: " << cycle_counter_ << std::endl;
                #endif
            }
            
            if (cycle_counter_ == 0) {
                busy_ = false;
                #ifdef DEBUG_PE
                std::cout << "PE: computation complete" << std::endl;
                #endif
            }
        }
        
        // Schedule next tick using UniqueEvent
        tick_event_.schedule(1);
    }
    
} // namespace gemmini 