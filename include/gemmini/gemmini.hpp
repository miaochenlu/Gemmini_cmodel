// gemmini.hpp - Top-level for Gemmini simulator using SPARTA framework
#pragma once

#include "sparta/app/Simulation.hpp"
#include "sparta/app/CommandLineSimulator.hpp"
#include "sparta/app/SimulationConfiguration.hpp"

#include "gemmini/matrix_multiplier.hpp"

namespace gemmini
{
    // GemminiSimulation - Top level simulation for the Gemmini systolic array accelerator
    class GemminiSimulation : public sparta::app::Simulation
    {
    public:
        // Constructor
        GemminiSimulation(sparta::Scheduler* scheduler);
        
        // Destructor
        ~GemminiSimulation();
        
        // Run simulation with input matrices
        void runSimulation(const MatrixPtr& matrix_a, const MatrixPtr& matrix_b);
        
    private:
        // Implementation of pure virtual methods from Simulation
        virtual void buildTree_() override;
        virtual void configureTree_() override;
        virtual void bindTree_() override;
        
        // Matrix multiplier resource
        MatrixMultiplier* matrix_multiplier_ = nullptr;
    };
    
} // namespace gemmini 