# Gemmini Processing Element (PE) Design Document

## Overview

The Processing Element (PE) is a fundamental building block of the Gemmini systolic array architecture. Each PE performs multiply-accumulate (MAC) operations essential for matrix multiplication and convolutional neural network computations.

## Module Structure

### Class Definition

The PE is implemented as a C++ class that inherits from `sparta::Unit`, providing integration with the SPARTA simulation framework.

Key components:

- **Port Interface**: Manages input/output connections
- **Event Management**: Handles timing and scheduling
- **Data Processing**: Performs the actual MAC operations
- **Control Logic**: Manages PE states and operations

### State Variables

- `weight_`: Stores the weight value (typically from weight matrix)
- `input_`: Stores the input data value (typically from activation matrix)
- `output_`: Value forwarded to the next PE in the array
- `accumulator_`: Accumulates the results of MAC operations
- `busy_`: Indicates whether PE is currently processing
- `cycle_counter_`: Tracks remaining cycles for current operation

### Parameters

- `compute_time_`: Number of cycles required for MAC operation
- `data_width_`: Bit width of the data processed by the PE

## Data Flow

### Data Flow Diagram

```
                    +----------------------+
                    |                      |
 Weight Input  ---->|                      |
 (in_weight)        |                      |
                    |                      |
                    |        PE            |----> Data Output
 Data Input    ---->|                      |      (out_data)
 (in_data)          |   [accumulator]      |
                    |                      |
                    |                      |----> Result Output
 Control Input ---->|                      |      (out_result)
 (in_control)       |                      |
                    +----------------------+
```

### Input Paths

1. **Weight Input** (`in_weight`): Receives weight values from the weight buffer
2. **Data Input** (`in_data`): Receives activation data from previous PE or input buffer
3. **Control Input** (`in_control`): Receives control signals for operation management

### Output Paths

1. **Data Output** (`out_data`): Forwards input data to the next PE in the array
2. **Result Output** (`out_result`): Sends accumulated result when requested

### Internal Data Flow

1. Weight is stored in the PE
2. Input data arrives and is stored
3. MAC operation is performed: accumulator += weight \* input
4. Input data is forwarded to the next PE
5. Result is available through the result output port when requested

## Timing and Cycles

### Operation Timing

- Each MAC operation takes `compute_time_` cycles to complete
- The PE sets `busy_` flag during computation
- `cycle_counter_` decrements each clock cycle until reaching 0

### Processing States

1. **Idle State**: Ready to receive new input data
2. **Busy State**: Currently processing a MAC operation
3. **Output State**: Sending accumulated result (when requested)

### Cycle-by-Cycle Operation

1. **Cycle 0**: Receive input data and weight
2. **Cycle 1 to compute*time***: Process MAC operation
3. **After compute*time* cycles**: Return to idle state

### Timing Diagram

```
Clock       : |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
             : |-----|-----|-----|-----|-----|-----|-----|-----|
Weight      : |  W  |     |     |     |     |     |     |     |
             : |-----|-----|-----|-----|-----|-----|-----|-----|
Data Input  : |  D  |     |     |     |     |  D' |     |     |
             : |-----|-----|-----|-----|-----|-----|-----|-----|
Data Output : |     |  D  |     |     |     |     |  D' |     |
             : |-----|-----|-----|-----|-----|-----|-----|-----|
Busy        : |     |  X  |  X  |  X  |     |     |  X  |  X  |
             : |-----|-----|-----|-----|-----|-----|-----|-----|
             :           ^           ^           ^           ^
             :           |           |           |           |
             :       MAC starts   MAC ends   MAC starts   MAC ends
             :       cycle_counter=3          cycle_counter=3
```

_(Assumes compute*time*=3, W=weight, D/D'=different data inputs, X=busy)_

## Control Signals

- **Reset Signal (1)**: Resets the accumulator to 0
- **Read Signal (2)**: Requests the PE to output its accumulated result

## Statistics

- `total_macs_`: Counts the number of MAC operations performed

## Design Considerations

1. **Pipelining**: The PE forwards input data immediately while computation occurs in parallel
2. **Busy Handling**: Ignores new input data when busy with computation
3. **Parametrized Timing**: Configurable compute time allows modeling different hardware implementations
4. **Fixed-Point Arithmetic**: Uses int16_t for inputs and int32_t for accumulation to prevent overflow

## Integration in Systolic Array

In a complete systolic array:

- PEs are arranged in a grid
- Weights flow from top to bottom
- Activation data flows from left to right
- Results accumulate within each PE
- Final results are read out when computation is complete

### Systolic Array Arrangement

```
        Data Flow →
      +----+----+----+----+
      | PE | PE | PE | PE |
      +----+----+----+----+
 W    | PE | PE | PE | PE |
 e    +----+----+----+----+
 i    | PE | PE | PE | PE |
 g    +----+----+----+----+
 h    | PE | PE | PE | PE |
 t    +----+----+----+----+
 ↓
```
