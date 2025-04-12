// pe.hpp - Processing Element for Gemmini Systolic Array using SPARTA framework
#pragma once

#include <cstdint>
#include <iostream>

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

namespace gemmini
{
    class PE;

    // Parameter Set for PE
    class PEParameterSet : public sparta::ParameterSet
    {
    public:
        // Constructor - connect params to the PE's TreeNode
        PEParameterSet(sparta::TreeNode* n) :
            sparta::ParameterSet(n)
        {
            // Parameters are initialized using the PARAMETER macro
        }

        // Parameters
        PARAMETER(uint32_t, compute_time, 1, "Cycles required for MAC operation")
        PARAMETER(uint32_t, data_width, 16, "Data width in bits")
    };

    // Port Set for PE
    class PEPortSet : public sparta::PortSet
    {
    public:
        // Constructor
        PEPortSet(sparta::TreeNode* n) :
            sparta::PortSet(n),
            
            // Data input ports
            in_weight(n, "in_weight", sparta::SchedulingPhase::Tick, 0),
            in_data(n, "in_data", sparta::SchedulingPhase::Tick, 0),
            in_control(n, "in_control", sparta::SchedulingPhase::Tick, 0),
            
            // Data output ports
            out_data(n, "out_data"),
            out_result(n, "out_result")
        {
            // No need to register ports explicitly - the base class does this
        }

        // Input ports - data arrives to these ports from outside
        sparta::DataInPort<int16_t> in_weight;
        sparta::DataInPort<int16_t> in_data;
        sparta::DataInPort<uint32_t> in_control;
        
        // Output ports - data sent from these ports to outside
        sparta::DataOutPort<int16_t> out_data;
        sparta::DataOutPort<int32_t> out_result;
    };

    // Processing Element class - basic compute unit for the systolic array using SPARTA
    class PE : public sparta::Unit
    {
    public:
        // Static name for this resource
        static const char name[];
        
        // Constructor
        PE(sparta::TreeNode* node, const PEParameterSet* params);
        
        // Define parameter set type for use with ResourceFactory
        typedef PEParameterSet ParameterSet;
        
        // Factory for PE creation
        class Factory : public sparta::ResourceFactory<PE, PEParameterSet>
        {
        public:
            // Using parent constructor
            using sparta::ResourceFactory<PE, PEParameterSet>::ResourceFactory;
        };
        
        // Return port set
        PEPortSet& getPortSet() { return port_set_; }
        
        // Direct access methods
        void setWeight(int16_t weight);
        void receiveData(int16_t data);
        void controlSignal(uint32_t signal);
        
    private:
        // Port set
        PEPortSet port_set_;
        
        // Event set for scheduling
        sparta::EventSet unit_event_set_;
        
        // Logger
        sparta::log::MessageSource logger_;
        
        // Internal state
        int16_t weight_ = 0;     // Weight stored in PE
        int16_t input_ = 0;      // Current input value
        int16_t output_ = 0;     // Output value
        int32_t accumulator_ = 0; // Accumulated result
        bool busy_ = false;      // Busy status
        uint32_t cycle_counter_ = 0; // Cycles remaining for computation
        
        // Configuration from parameters
        const uint32_t compute_time_;
        const uint32_t data_width_;
        
        // Statistics
        sparta::Counter total_macs_; // Count of MAC operations
        
        // Tick event
        sparta::UniqueEvent<> tick_event_;
        
        // Internal methods
        void handleWeight_(const int16_t& weight);
        void handleData_(const int16_t& data);
        void handleControl_(const uint32_t& signal);
        void computeResult_();
        void tick_();
    };
    
} // namespace gemmini 