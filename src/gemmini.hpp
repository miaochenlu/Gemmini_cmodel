// gemmini.hpp - Top-level for Gemmini simulator using SPARTA framework
#pragma once

#include "sparta/app/Simulation.hpp"
#include "sparta/app/CommandLineSimulator.hpp"
#include "sparta/app/SimulationConfiguration.hpp"

#include "gemmini/common.hpp"
#include "gemmini/matrix_multiplier.hpp"

BEGIN_NS(gemmini)

// GemminiSimulation - Top level simulation for the Gemmini systolic array accelerator
class GemminiSimulation : public sparta::app::Simulation {
public:
    // Constructor
    GemminiSimulation(sparta::Scheduler* scheduler);

    // Destructor
    ~GemminiSimulation();

    // Run simulation with input matrices
    void RunSimulation(const MatrixPtr & matrixA, const MatrixPtr & matrixB);

private:
    // Implementation of pure virtual methods from Simulation
    virtual void buildTree_() override;
    virtual void configureTree_() override;
    virtual void bindTree_() override;

    // Matrix multiplier resource
    MatrixMultiplier* mMatrixMultiplier = nullptr;
};

END_NS(gemmini)