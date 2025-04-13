# 基于 SPARTA 框架开发性能模型的详细指南

SPARTA 是一个强大的架构模拟框架，用于构建周期精确的性能模型。以下是开发基于 SPARTA 的性能模型的详细步骤和经验，重点关注 Gemmini systolic array 的实现。

## 1. 基本结构和类层次

SPARTA 模型由以下核心组件构成：

1. **Unit 类**: 所有计算单元的基类
2. **TreeNode**: 组织单元之间层次关系
3. **Port**: 单元间通信接口
4. **Event**: 用于调度和处理事件
5. **Parameter**: 可配置的模型参数

一个典型的 SPARTA 模型结构如下：

```
顶层模拟类 (GemminiSimulation)
  |
  ├── 主要组件 (MatrixMultiplier)
  |     |
  |     ├── 子组件 (SystolicArray)
  |     |     |
  |     |     └── 基本处理单元 (PE)
  |     |
  |     └── 其他组件
  |
  └── 其他组件
```

## 2. 详细实现步骤

### 2.1 定义处理单元(PE)

处理单元是模型的基本计算单位。以下是实现步骤：

```cpp
// 1. 定义参数集
class PEParameterSet : public sparta::ParameterSet {
public:
    PEParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) { }

    // 声明参数
    PARAMETER(uint32_t, compute_cycles, 1, "MAC操作所需周期数")
    PARAMETER(uint32_t, data_width, 16, "数据位宽")
};

// 2. 定义端口集
class PEPortSet : public sparta::PortSet {
public:
    PEPortSet(sparta::TreeNode* n) : sparta::PortSet(n),
        // 输入端口
        in_weight(n, "in_weight", sparta::SchedulingPhase::Tick, 0),
        in_data(n, "in_data", sparta::SchedulingPhase::Tick, 0),
        in_control(n, "in_control", sparta::SchedulingPhase::Tick, 0),

        // 输出端口
        out_data(n, "out_data"),
        out_result(n, "out_result")
    { }

    // 声明端口
    sparta::DataInPort<int16_t> in_weight;    // 权重输入
    sparta::DataInPort<int16_t> in_data;      // 数据输入
    sparta::DataInPort<uint32_t> in_control;  // 控制信号

    sparta::DataOutPort<int16_t> out_data;    // 转发数据
    sparta::DataOutPort<int32_t> out_result;  // 结果输出
};

// 3. 定义处理单元类
class PE : public sparta::Unit {
public:
    static const char name[];  // 单元名称

    // 构造函数
    PE(sparta::TreeNode* node, const PEParameterSet* params);

    // 工厂类定义
    typedef PEParameterSet ParameterSet;
    class Factory : public sparta::ResourceFactory<PE, PEParameterSet> {
    public:
        using sparta::ResourceFactory<PE, PEParameterSet>::ResourceFactory;
    };

    // 获取端口集
    PEPortSet& getPortSet() { return port_set_; }

    // 直接访问方法
    void setWeight(int16_t weight);
    void receiveData(int16_t data);
    void controlSignal(uint32_t signal);

private:
    // 端口集
    PEPortSet port_set_;

    // 事件集
    sparta::EventSet unit_event_set_;

    // 日志
    sparta::log::MessageSource logger_;

    // 内部状态
    int16_t weight_ = 0;        // 存储权重
    int16_t input_ = 0;         // 输入数据
    int16_t output_ = 0;        // 输出数据
    int32_t accumulator_ = 0;   // 累加器
    bool busy_ = false;         // 忙状态
    uint32_t cycle_counter_ = 0; // 计算周期计数器

    // 配置参数
    const uint32_t compute_cycles_;
    const uint32_t data_width_;

    // 统计信息
    sparta::Counter total_macs_; // MAC操作计数

    // Tick事件
    sparta::UniqueEvent<> tick_event_;

    // 内部方法
    void handleWeight_(const int16_t& weight);   // 处理权重输入
    void handleData_(const int16_t& data);       // 处理数据输入
    void handleControl_(const uint32_t& signal); // 处理控制信号
    void computeResult_();                       // 计算MAC结果
    void tick_();                                // 时钟周期处理
};
```

### 2.2 PE 实现

```cpp
// 静态名称初始化
const char PE::name[] = "pe";

// 构造函数实现
PE::PE(sparta::TreeNode* node, const PEParameterSet* params) :
    sparta::Unit(node),
    port_set_(node),
    unit_event_set_(node),
    logger_(node, "pe", "Processing Element Log"),
    compute_cycles_(params->compute_cycles),
    data_width_(params->data_width),
    total_macs_(getStatisticSet(), "total_macs", "Count of MAC operations", sparta::Counter::COUNT_NORMAL),
    tick_event_(&unit_event_set_, "tick_event", CREATE_SPARTA_HANDLER(PE, tick_))
{
    // 注册端口处理器
    port_set_.in_weight.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(PE, handleWeight_, int16_t));
    port_set_.in_data.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(PE, handleData_, int16_t));
    port_set_.in_control.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(PE, handleControl_, uint32_t));

    // 创建并注册tick事件
    sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(PE, tick_));
}

// 权重处理
void PE::handleWeight_(const int16_t& weight) {
    weight_ = weight;
    // 日志记录
    std::cout << "PE: Received weight: " << weight << std::endl;
}

// 数据处理
void PE::handleData_(const int16_t& data) {
    if (busy_) {
        std::cout << "PE: Busy, ignoring data: " << data << std::endl;
        return;
    }

    input_ = data;
    output_ = input_;  // 转发数据

    // 发送数据到下一个PE
    port_set_.out_data.send(output_);

    // 开始MAC计算
    computeResult_();
}

// MAC计算实现
void PE::computeResult_() {
    busy_ = true;
    cycle_counter_ = compute_cycles_;

    // 执行MAC操作
    int32_t product = static_cast<int32_t>(weight_) * static_cast<int32_t>(input_);
    accumulator_ += product;

    // 更新统计信息
    total_macs_++;

    std::cout << "PE: MAC: " << weight_ << " * " << input_
              << " = " << product
              << ", acc = " << accumulator_ << std::endl;
}

// 时钟周期处理
void PE::tick_() {
    if (busy_) {
        if (cycle_counter_ > 0) {
            --cycle_counter_;
        }

        if (cycle_counter_ == 0) {
            busy_ = false;
        }
    }

    // 调度下一个时钟周期
    tick_event_.schedule(1);
}
```

### 2.3 定义 Systolic Array

```cpp
// 1. 定义参数集
class SystolicArrayParameterSet : public sparta::ParameterSet {
public:
    SystolicArrayParameterSet(sparta::TreeNode* n) : sparta::ParameterSet(n) { }

    // 声明参数
    PARAMETER(uint32_t, rows, 4, "阵列行数")
    PARAMETER(uint32_t, cols, 4, "阵列列数")
    PARAMETER(uint32_t, compute_cycles, 1, "PE MAC操作所需周期数")
};

// 2. 定义端口集
class SystolicArrayPortSet : public sparta::PortSet {
public:
    SystolicArrayPortSet(sparta::TreeNode* n) : sparta::PortSet(n),
        // 输入端口
        in_weights(n, "in_weights", sparta::SchedulingPhase::Tick, 0),
        in_vector(n, "in_vector", sparta::SchedulingPhase::Tick, 0),
        in_control(n, "in_control", sparta::SchedulingPhase::Tick, 0),

        // 输出端口
        out_results(n, "out_results")
    { }

    // 声明端口
    sparta::DataInPort<MatrixPtr> in_weights;  // 权重矩阵输入
    sparta::DataInPort<VectorPtr> in_vector;   // 输入向量
    sparta::DataInPort<uint32_t> in_control;   // 控制信号

    sparta::DataOutPort<MatrixPtr> out_results; // 结果输出
};

// 3. 定义Systolic Array类
class SystolicArray : public sparta::Unit {
public:
    static const char name[];

    SystolicArray(sparta::TreeNode* node, const SystolicArrayParameterSet* params);

    typedef SystolicArrayParameterSet ParameterSet;
    class Factory : public sparta::ResourceFactory<SystolicArray, SystolicArrayParameterSet> {
    public:
        using sparta::ResourceFactory<SystolicArray, SystolicArrayParameterSet>::ResourceFactory;
    };

    SystolicArrayPortSet& getPortSet() { return port_set_; }

private:
    // 端口集
    SystolicArrayPortSet port_set_;

    // 事件集
    sparta::EventSet unit_event_set_;

    // 日志
    sparta::log::MessageSource logger_;

    // 配置参数
    const uint32_t rows_;
    const uint32_t cols_;
    const uint32_t compute_cycles_;

    // PE阵列
    std::vector<PE*> pes_; // 扁平化的2D阵列

    // 当前状态
    bool processing_ = false;
    uint32_t current_cycle_ = 0;
    uint32_t total_cycles_needed_ = 0;
    VectorPtr current_input_;
    MatrixPtr result_matrix_;

    // 统计信息
    sparta::Counter total_matrix_ops_;

    // Tick事件
    sparta::UniqueEvent<> tick_event_;

    // 内部方法
    void handleWeights_(const MatrixPtr& weights);
    void handleVector_(const VectorPtr& input);
    void handleControl_(const uint32_t& signal);

    void processOneCycle_();
    void computationComplete_();
    void tick_();

    // 辅助方法
    PE* getPE(uint32_t row, uint32_t col) { return pes_[row * cols_ + col]; }
    std::string getPEName(uint32_t row, uint32_t col) const {
        return "pe_" + std::to_string(row) + "_" + std::to_string(col);
    }
};
```

### 2.4 Systolic Array 实现

```cpp
// 静态名称初始化
const char SystolicArray::name[] = "systolic_array";

// 构造函数实现
SystolicArray::SystolicArray(sparta::TreeNode* node, const SystolicArrayParameterSet* params) :
    sparta::Unit(node),
    port_set_(node),
    unit_event_set_(node),
    logger_(node, "systolic_array", "Systolic Array Log"),
    rows_(params->rows),
    cols_(params->cols),
    compute_cycles_(params->compute_cycles),
    total_matrix_ops_(getStatisticSet(), "total_matrix_ops", "Count of matrix operations", sparta::Counter::COUNT_NORMAL),
    tick_event_(&unit_event_set_, "tick_event", CREATE_SPARTA_HANDLER(SystolicArray, tick_))
{
    // 注册端口处理器
    port_set_.in_weights.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, handleWeights_, MatrixPtr));
    port_set_.in_vector.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, handleVector_, VectorPtr));
    port_set_.in_control.registerConsumerHandler(CREATE_SPARTA_HANDLER_WITH_DATA(SystolicArray, handleControl_, uint32_t));

    // 创建4x4 PE阵列
    std::cout << "Creating " << rows_ << "x" << cols_ << " systolic array" << std::endl;

    for (uint32_t r = 0; r < rows_; ++r) {
        for (uint32_t c = 0; c < cols_; ++c) {
            // 创建PE节点
            std::string pe_name = getPEName(r, c);
            std::string pe_desc = "PE at (" + std::to_string(r) + "," + std::to_string(c) + ")";
            sparta::TreeNode* pe_node = new sparta::TreeNode(node, pe_name, pe_desc);

            // 创建PE参数集
            auto pe_params = new PEParameterSet(pe_node);
            pe_params->compute_cycles = compute_cycles_;

            // 创建PE
            PE::Factory pe_factory;
            PE* pe = static_cast<PE*>(pe_factory.createResource(pe_node, pe_params));

            // 存储PE指针
            pes_.push_back(pe);

            std::cout << "Created PE at (" << r << "," << c << ")" << std::endl;
        }
    }

    // 连接PE阵列 (数据水平流动，结果在PE中累积)
    std::cout << "Connecting PEs in the systolic array" << std::endl;

    for (uint32_t r = 0; r < rows_; ++r) {
        for (uint32_t c = 0; c < cols_; ++c) {
            PE* current_pe = getPE(r, c);

            // 连接到右侧PE (如果不是最后一列)
            if (c < cols_ - 1) {
                PE* right_pe = getPE(r, c + 1);
                current_pe->getPortSet().out_data.bind(right_pe->getPortSet().in_data);
                std::cout << "Connected PE(" << r << "," << c << ") -> PE(" << r << "," << (c+1) << ")" << std::endl;
            }
        }
    }

    // 初始化结果矩阵
    result_matrix_ = create_matrix_ptr<Matrix>(rows_, 1);

    // 创建并注册tick事件
    sparta::StartupEvent(node, CREATE_SPARTA_HANDLER(SystolicArray, tick_));

    std::cout << "Systolic Array initialization complete" << std::endl;
}

// 处理权重矩阵
void SystolicArray::handleWeights_(const MatrixPtr& weights) {
    // 检查权重矩阵尺寸
    if (weights->rows() != rows_ || weights->cols() != cols_) {
        std::cerr << "Error: Weight matrix dimensions mismatch" << std::endl;
        return;
    }

    std::cout << "Loading weights into PEs" << std::endl;

    // 加载权重到每个PE
    for (uint32_t r = 0; r < rows_; ++r) {
        for (uint32_t c = 0; c < cols_; ++c) {
            PE* pe = getPE(r, c);
            pe->setWeight(weights->at(r, c));
        }
    }
}

// 处理输入向量
void SystolicArray::handleVector_(const VectorPtr& input) {
    if (processing_) {
        std::cerr << "Error: Still processing previous input" << std::endl;
        return;
    }

    // 检查输入向量尺寸
    if (input->size() != rows_) {
        std::cerr << "Error: Input vector size mismatch" << std::endl;
        return;
    }

    std::cout << "Starting processing of input vector" << std::endl;

    // 重置结果矩阵
    result_matrix_ = create_matrix_ptr<Matrix>(rows_, 1);

    // 重置每个PE的累加器
    for (uint32_t r = 0; r < rows_; ++r) {
        for (uint32_t c = 0; c < cols_; ++c) {
            getPE(r, c)->controlSignal(1); // 重置信号
        }
    }

    // 保存输入以供处理
    current_input_ = input;

    // 开始处理
    processing_ = true;
    current_cycle_ = 0;

    // 计算所需总周期 (对于倾斜调度)
    total_cycles_needed_ = rows_ + cols_ - 1 + compute_cycles_;

    std::cout << "Processing will take " << total_cycles_needed_ << " cycles" << std::endl;
}

// 处理一个周期
void SystolicArray::processOneCycle_() {
    if (!processing_) {
        return;
    }

    std::cout << "Processing cycle " << current_cycle_ << " of " << total_cycles_needed_ << std::endl;

    // 实现对角线波前数据流
    for (uint32_t r = 0; r < rows_; ++r) {
        // 计算本周期哪一列应该接收数据 (基于倾斜调度)
        int32_t active_col = current_cycle_ - r;

        if (active_col >= 0 && active_col < static_cast<int32_t>(cols_)) {
            PE* pe = getPE(r, active_col);
            uint32_t input_idx = r;

            if (input_idx < current_input_->size()) {
                pe->receiveData((*current_input_)[input_idx]);
                std::cout << "Sent input " << (*current_input_)[input_idx] << " to PE(" << r << "," << active_col << ")" << std::endl;
            }
        }
    }

    // 增加周期计数器
    current_cycle_++;

    // 检查处理是否完成
    if (current_cycle_ >= total_cycles_needed_) {
        computationComplete_();
    }
}

// 计算完成
void SystolicArray::computationComplete_() {
    std::cout << "Computation complete, collecting results" << std::endl;

    // 从最右列的PE收集结果
    for (uint32_t r = 0; r < rows_; ++r) {
        PE* pe = getPE(r, cols_ - 1);
        pe->controlSignal(2); // 读取信号

        // 注意: 在实际实现中，需要异步处理PE的响应
        // 这里简化处理
        result_matrix_->at(r, 0) = 0; // 这里应该替换为实际结果
    }

    // 发送结果矩阵
    port_set_.out_results.send(result_matrix_);

    // 更新统计信息
    total_matrix_ops_++;

    // 重置处理标志
    processing_ = false;
}

// 时钟周期处理
void SystolicArray::tick_() {
    // 处理一个周期
    processOneCycle_();

    // 调度下一个周期
    tick_event_.schedule(1);
}
```

### 2.5 矩阵乘法器

```cpp
class MatrixMultiplier : public sparta::Unit {
public:
    static const char name[];

    MatrixMultiplier(sparta::TreeNode* node, const MatrixMultiplierParameterSet* params);

    typedef MatrixMultiplierParameterSet ParameterSet;
    class Factory : public sparta::ResourceFactory<MatrixMultiplier, MatrixMultiplierParameterSet> {
    public:
        using sparta::ResourceFactory<MatrixMultiplier, MatrixMultiplierParameterSet>::ResourceFactory;
    };

    // 直接访问方法
    void multiply(const MatrixPtr& a, const MatrixPtr& b);
    MatrixPtr getResult() const { return result_matrix_; }

private:
    // 端口和内部状态 (类似于SystolicArray)
    // ...

    // 内部方法
    void startMultiplication_();
    void processNextBlock_();
    void multiplierDone_();
};
```

### 2.6 顶层模拟

```cpp
class GemminiSimulation : public sparta::app::Simulation {
public:
    GemminiSimulation(sparta::Scheduler* scheduler);
    ~GemminiSimulation();

    void runSimulation(const MatrixPtr& matrix_a, const MatrixPtr& matrix_b);

private:
    // 实现sparta::app::Simulation的纯虚函数
    virtual void buildTree_() override;
    virtual void configureTree_() override;
    virtual void bindTree_() override;

    // 矩阵乘法器
    MatrixMultiplier* matrix_multiplier_ = nullptr;
};

// 实现
GemminiSimulation::GemminiSimulation(sparta::Scheduler* scheduler) :
    sparta::app::Simulation("GemminiSim", scheduler)
{
}

GemminiSimulation::~GemminiSimulation() {
    getRoot()->enterTeardown();
}

void GemminiSimulation::buildTree_() {
    // 创建根节点
    auto root_node = getRoot();

    // 创建矩阵乘法器节点
    sparta::TreeNode* mm_node = new sparta::TreeNode(root_node, "matrix_multiplier", "Matrix Multiplier");

    // 创建矩阵乘法器参数集
    auto mm_params = new MatrixMultiplier::ParameterSet(mm_node);

    // 创建矩阵乘法器
    MatrixMultiplier::Factory mm_factory;
    matrix_multiplier_ = static_cast<MatrixMultiplier*>(mm_factory.createResource(mm_node, mm_params));
}

void GemminiSimulation::configureTree_() {
    // 配置模拟树(如果需要)
}

void GemminiSimulation::bindTree_() {
    // 绑定模拟树组件(如果需要)
}

void GemminiSimulation::runSimulation(const MatrixPtr& matrix_a, const MatrixPtr& matrix_b) {
    std::cout << "Starting Gemmini matrix multiplication simulation..." << std::endl;

    // 打印矩阵尺寸
    std::cout << "Matrix A: " << matrix_a->rows() << "x" << matrix_a->cols() << std::endl;
    std::cout << "Matrix B: " << matrix_b->rows() << "x" << matrix_b->cols() << std::endl;

    // 计算预期周期
    uint32_t expected_cycles = calculateExpectedCycles(matrix_a, matrix_b);
    std::cout << "Expected simulation time: " << expected_cycles << " cycles" << std::endl;

    // 执行矩阵乘法
    matrix_multiplier_->multiply(matrix_a, matrix_b);

    // 运行模拟
    runRaw(expected_cycles);

    // 获取结果
    MatrixPtr result = matrix_multiplier_->getResult();

    // 打印结果
    std::cout << "Matrix multiplication result:" << std::endl;
    for (uint32_t r = 0; r < result->rows(); ++r) {
        std::cout << "  [";
        for (uint32_t c = 0; c < result->cols(); ++c) {
            std::cout << result->at(r, c);
            if (c < result->cols() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}
```

## 3. 关键架构模式

### 3.1 权重固定(Weight Stationary)设计

在我们实现的 Gemmini systolic array 中，采用了权重固定的设计模式：

1. 权重预先加载到每个 PE 中并保持不变
2. 输入数据从左侧进入并向右流动
3. 各 PE 执行 MAC 操作并累积结果
4. 最终结果从最右列 PE 读取

### 3.2 倾斜数据流(Skewed Data Flow)

为了实现正确的乘法计算，输入数据需要以对角线波前模式进入阵列：

```
Cycle 0:  d00 --- --- ---
           |
Cycle 1:  d10 d01 --- ---
           |    |
Cycle 2:  d20 d11 d02 ---
           |    |    |
Cycle 3:  d30 d21 d12 d03
                |    |
Cycle 4:  ---  d31 d22 d13
                     |    |
Cycle 5:  ---  ---  d32 d23
                          |
Cycle 6:  ---  ---  ---  d33
```

这种倾斜调度确保了正确的数据对齐，使得每个 PE 的输入数据能够与正确的权重相乘。

## 4. 测试方法

测试代码应该包含以下功能：

1. 创建随机测试矩阵和向量
2. 计算预期结果作为参考
3. 运行模拟并比较结果

```cpp
// 1. 创建测试矩阵和向量
MatrixPtr test_matrix = createTestMatrix(4, 4, 1, 5);
VectorPtr test_vector = createTestVector(4, 1, 5);

// 2. 计算预期结果
MatrixPtr expected_result = calculateExpectedResult(test_matrix, test_vector);

// 3. 打印测试数据
printMatrix(test_matrix, "Weight Matrix");
printVector(test_vector, "Input Vector");
printMatrix(expected_result, "Expected Result");

// 4. 运行模拟
testMatrixMultiplication();
```

## 5. 性能分析和优化

使用 SPARTA 框架的统计和计数器功能来收集性能指标：

```cpp
// 在PE类中定义计数器
sparta::Counter total_macs_; // MAC操作计数

// 在构造函数中初始化
total_macs_(getStatisticSet(), "total_macs", "Count of MAC operations", sparta::Counter::COUNT_NORMAL)

// 在执行MAC操作时更新
void PE::computeResult_() {
    // 执行MAC操作
    int32_t product = static_cast<int32_t>(weight_) * static_cast<int32_t>(input_);
    accumulator_ += product;

    // 更新统计信息
    total_macs_++;
}
```

## 6. 扩展和改进方向

1. **支持更大矩阵**: 通过分块处理支持大于 4x4 的矩阵
2. **支持不同数据类型**: 添加对 int8, fp16 等数据类型的支持
3. **实现不同的数据流模式**: 实现输出固定或输入固定模式
4. **添加内存接口**: 实现与内存系统的交互
5. **添加 DMA 控制器**: 支持数据的自动加载和卸载

## 总结

基于 SPARTA 框架开发性能模型的关键步骤包括：

1. 定义基本计算单元(PE)及其参数、端口和内部状态
2. 组织计算单元形成层次结构(Systolic Array)
3. 实现数据流和计算逻辑
4. 定义顶层模拟接口
5. 开发测试代码并验证功能正确性

通过这种方法构建的模型可以准确模拟硬件加速器的行为和性能特性，为架构设计和优化提供有价值的见解。
