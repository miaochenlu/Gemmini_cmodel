// fifo.hpp - Configurable depth FIFO for delay modeling in Gemmini
#pragma once

#include <cstdint>
#include <queue>
#include <deque>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "sparta/events/EventSet.hpp"
#include "sparta/events/UniqueEvent.hpp"
#include "sparta/log/MessageSource.hpp"
#include "sparta/ports/DataPort.hpp"
#include "sparta/ports/PortSet.hpp"
#include "sparta/simulation/ParameterSet.hpp"
#include "sparta/simulation/ResourceFactory.hpp"
#include "sparta/simulation/TreeNode.hpp"
#include "sparta/simulation/Unit.hpp"

#include "gemmini/common.hpp"

BEGIN_NS(gemmini)

// Forward declaration
template <typename T>
class DelayFifo;

// Helper class to generate resource name based on type
template <typename T>
struct DelayFifoNameHelper {
    static std::string GetName() {
        if constexpr (std::is_same_v<T, int16_t>) {
            return "delay_fifo_int16";
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return "delay_fifo_int32";
        } else if constexpr (std::is_same_v<T, float>) {
            return "delay_fifo_float";
        } else if constexpr (std::is_same_v<T, double>) {
            return "delay_fifo_double";
        } else if constexpr (std::is_same_v<T, int>) {
            return "delay_fifo_int";
        } else {
            return "delay_fifo_unknown";
        }
    }
};

// Parameter Set for DelayFifo
template <typename T>
class DelayFifoParameterSet : public sparta::ParameterSet {
public:
    // Constructor - connect params to the DelayFifo's TreeNode
    DelayFifoParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) {}

    // Parameters
    PARAMETER(uint32_t, depth, 1, "Depth of the FIFO (number of cycles of delay)")
    PARAMETER(bool, debug_mode, false, "Enable debug output")
};

// Port Set for DelayFifo
template <typename T>
class DelayFifoPortSet : public sparta::PortSet {
public:
    // Constructor
    DelayFifoPortSet(sparta::TreeNode* n)
        : sparta::PortSet(n),
          // Data input port
          in(n, "in", sparta::SchedulingPhase::Tick, 0),
          // Data output port
          out(n, "out") {
        // No need to register ports explicitly - the base class does this
    }

    // Input port - data arrives to this port
    sparta::DataInPort<T> in;
    
    // Output port - data exits from this port
    sparta::DataOutPort<T> out;
};

// DelayFifo class - models a FIFO with configurable delay
template <typename T>
class DelayFifo : public sparta::Unit {
public:
    // Static name required by sparta Resource
    static const char* name;

    // Constructor
    DelayFifo(sparta::TreeNode* node, const DelayFifoParameterSet<T>* params)
        : sparta::Unit(node), mPortSet(node), mUnitEventSet(node),
          mLogger(node, "delay_fifo", "Delay FIFO Log"),
          mDepth(params->depth), mDebugMode(params->debug_mode),
          mTickEvent(&mUnitEventSet, "tick_event", CREATE_SPARTA_HANDLER(DelayFifo, Tick)) {
        
        // Register port handlers
        mPortSet.in.registerConsumerHandler(
            CREATE_SPARTA_HANDLER_WITH_DATA(DelayFifo, HandleInput, T));
        
        // Create and register tick event
        sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(DelayFifo, Tick));
        
        if (mDebugMode) {
            std::cout << "DelayFifo created with depth " << mDepth << std::endl;
        }
    }
    
    // Resource name - defined using helper class
    static const char* getName() {
        static const std::string name = DelayFifoNameHelper<T>::GetName();
        return name.c_str();
    }
    
    // Define parameter set type for use with ResourceFactory
    typedef DelayFifoParameterSet<T> ParameterSet;
    
    // Factory for DelayFifo creation
    class Factory : public sparta::ResourceFactory<DelayFifo<T>, DelayFifoParameterSet<T>> {
    public:
        // Using parent constructor
        using sparta::ResourceFactory<DelayFifo<T>, DelayFifoParameterSet<T>>::ResourceFactory;
        
        // Method to provide resource name (removed override keyword)
        const char* getResourceName() const {
            return DelayFifo<T>::getName();
        }
    };
    
    // Return port set
    DelayFifoPortSet<T>& GetPortSet() { return mPortSet; }
    
    // Direct methods to push data (for testing)
    void Push(const T& data) {
        if (mDebugMode) {
            std::cout << "DelayFifo: Pushing data: " << data << std::endl;
        }
        
        HandleInput(data);
    }
    
private:
    // Port set
    DelayFifoPortSet<T> mPortSet;
    
    // Event set for scheduling
    sparta::EventSet mUnitEventSet;
    
    // Logger
    sparta::log::MessageSource mLogger;
    
    // FIFO storage
    std::deque<T> mFifo;
    
    // Configuration
    const uint32_t mDepth;
    const bool mDebugMode;
    
    // Tick event for cycle-level simulation
    sparta::UniqueEvent<> mTickEvent;
    
    // Handle input data
    void HandleInput(const T& data) {
        // Push data into FIFO
        mFifo.push_back(data);
        
        if (mDebugMode) {
            std::cout << "DelayFifo: Received data: " << data << " (FIFO size: " << mFifo.size() << ")" << std::endl;
        }
    }
    
    // Process FIFO on each cycle
    void Tick() {
        // If FIFO has accumulated enough data (equal to or greater than depth), pop from front
        if (mFifo.size() >= mDepth) {
            // Get data from front of FIFO
            T data = mFifo.front();
            mFifo.pop_front();
            
            // Send data to output port
            mPortSet.out.send(data);
            
            if (mDebugMode) {
                std::cout << "DelayFifo: Sending data: " << data << " after " << mDepth 
                          << " cycles (FIFO size: " << mFifo.size() << ")" << std::endl;
            }
        }
        
        // Schedule next tick
        mTickEvent.schedule(1);
    }
};

// Define static name member for each required type
template<> inline const char* DelayFifo<int16_t>::name = "delay_fifo_int16";
template<> inline const char* DelayFifo<int32_t>::name = "delay_fifo_int32";
template<> inline const char* DelayFifo<float>::name = "delay_fifo_float";
template<> inline const char* DelayFifo<double>::name = "delay_fifo_double";
// int is already covered by int32_t on most platforms

END_NS(gemmini) 