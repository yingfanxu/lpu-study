# Req\.077 LPUv1 RV32集成方案

# 一、背景与架构设计

## 1\.1 设计目标

LPU 当前没有 CPU，pymodel 生成的指令相当于手动展开、挨个执行。更好的方式是在 LPU 内部集成一个 RISC\-V 标量核，pymodel 中的算子代码用 C\+\+ 重写，通过开源 RISC\-V 编译器编译后，先在 Function Model 上运行通过，再在 LPU 的 RISC\-V 核上运行。

## 1\.2 RV\-LPU扩展指令规范

在标准RV核上拓展8个32位特殊寄存器VREG\[7:0\]用于存储指令信息，新增三条三条自定义指令，使用 custom\-0 主操作码（opcode\[6:0\] = 0x0B），统一采用 R\-type 格式：

|指令|func7|funct3|语义|
|---|---|---|---|
|lpu\_vstore|0x20|0|VREG\[X\[rs1\]\] = X\[rs2\]|
|lpu\_vload|0x20|1|X\[rd\] = VREG\[X\[rs1\]\]|
|lpu\_vdispatch|0x20|2|输出 \{VREG\[7\],\.\.\.,VREG\[0\]\} 并置位 valid|

lpu\_dispatch\_data\[255:0\] = \{VREG\[7\], VREG\[6\], VREG\[5\], VREG\[4\], VREG\[3\], VREG\[2\], VREG\[1\], VREG\[0\]\}。

调用dispatch指令将指令信息一并发送给IS。如果IS已满无法接收新指令，则反压住RV核。

我们不需要大改RV核，只要增加一个RV核到IS的数据通路；也不需要大改编译器，只要增加这三条自定义指令。

## 1\.3 同步机制

set\_flag / wait\_flag / fence 指令不需要额外 RISC\-V 自定义指令来支持。CPU 核只是把指令 dispatch 到 IS（Instruction Scheduler），IS 负责管理同步。具体来说：

- IS（su\_instr\_scheduler\.sv）对每个执行单元（GDMA/LDMA/MXU/ARU）维护独立的 FIFO 队列（instr\_q / wait\_q / set\_q / done\_q）

- set\_flag 和 wait\_flag 本身就是普通的 32 字节 LPU 指令，dispatch 进 IS 后 IS 自己处理 set\_wait\_table 的递增/递减

- 反压机制：每个单元的 recv\_rdy 信号在任意队列满时拉低，dispatch 端（RISC\-V 核）在那一拍停住即可

- CPU 如果被压住就暂停执行，等 IS 队列有空间后继续

因此 C\+\+ 算子代码中，set\_flag / wait\_flag 和 gdma\_gm2ub 地位完全相同——pack 好 256 位指令，lpu\_vstore × 8，lpu\_vdispatch 一发。不需要额外的 lpu\_wait 自定义指令或状态 CSR。

# 二、仓库结构

新增 2 个目录：

|目录|性质|功能|
|---|---|---|
|rv32sdk/<br>|新建，顶层目录|运行在 RISC\-V 核上的软件栈：intrinsic 头文件、ISA 编码层、C\+\+ 算子库、bare\-metal 启动代码、host\-mock 测试|
|cmodel/src/rv32/|新建，集成进 cmodel|RV32I\+M function model（解释器 \+ ELF loader \+ xlpuv 扩展 \+ dispatch callback）；作为 librv32sim\.a 被 runFuncTest 链接|

现有仓库不做结构性改动。cmodel/func\_tests/rv32/ 下新增的 \.cpp 测试文件会被已有的 file\(GLOB\_RECURSE \.\.\.\) 自动收录进 runFuncTest，不需要修改 CMakeLists\.txt。

# 三、开源依赖

|库|用途|地址|集成方式|
|---|---|---|---|
|ELFIO|cmodel/src/rv32/ 中加载 RV32 ELF32 文件|github\.com/serge1/ELFIO|纯 header\-only，放进 cmodel/third\_party/elfio/，零编译步骤|
|riscv\-gnu\-toolchain|把 rv32sdk/ 的 C\+\+ 算子编译成 rv32i ELF|github\.com/riscv\-collab/riscv\-gnu\-toolchain 或 apt install gcc\-riscv64\-unknown\-elf|外部工具链，不进仓库|
|GTest|cmodel 中已有，路径硬编码在 CMakeLists\.txt|—|不变|

不需要引入 Spike、LLVM 或任何 RISC\-V 模拟器库。

# 四、完整实施步骤

## 第 1 步 — 工具链验证

安装 riscv32 交叉编译器，写一个只有 \.insn r 0x0b, 0, 0x20, x0, %0, %1 的最小 C 文件，确认：

- 编译不报错（标准 GAS 支持 \.insn r，无需改编译器）

- objdump \-d 反汇编出 0x4000000B

```bash
# Ubuntu 快速安装
apt install gcc-riscv64-unknown-elf
# 用 -march=rv32im -mabi=ilp32 即可出 rv32
```

**验收标准：**能编译并反汇编出正确的机器码。

## 第 2 步 — xlpuv\.h \+ lpu\_isa\_types\.h \+ lpu\_isa\.h

**rv32sdk/include/xlpuv\.h**：规范里已有，原样复制。增加一个 LPU\_HOST\_MOCK 条件编译 guard，让 host 测试时替换成 C 函数而不是内联汇编：

```c
#ifdef LPU_HOST_MOCK
static uint32_t _vreg[8];
static void lpu_vstore(uint32_t idx, uint32_t val) { _vreg[idx & 7] = val; }
static void lpu_vdispatch(void) { /* 调 capture callback */ }
#else
// 规范里的原始 .insn r 版本
#endif
```

**rv32sdk/include/lpu\_isa\_types\.h**：参照 pymodel common\.py 的常量和 task\_gen\.py 的字段布局，定义 C\+\+ 中的编码辅助函数：

```cpp
// 小端 pack：把 val 的低 nbytes 字节写入 buf+offset
inline void pack_le(uint32_t words[8], int byte_offset, int nbytes, uint64_t val);

constexpr uint8_t OPCODE_GDMA_GM2UB  = 0;
constexpr uint8_t OPCODE_MXU_MATMUL  = 32;
constexpr uint8_t OPCODE_SET_FLAG    = 64;
constexpr uint8_t OPCODE_WAIT_FLAG   = 65;
constexpr uint8_t OPCODE_FENCE       = 66;
// ... 所有 opcode
```

**rv32sdk/include/lpu\_isa\.h**：对应 pymodel task\_gen\.py 里每个方法，每个方法 pack 完 32 字节，调 lpu\_vstore × 8 \+ lpu\_vdispatch：

```cpp
inline void gdma_gm2ub(uint32_t gm_addr, uint32_t ub_addr,
                        uint16_t tensor_m, uint16_t start_m,
                        uint16_t tensor_k1, uint16_t start_k1,
                        uint16_t tile_m, uint16_t tile_k1) {
    uint32_t w[8] = {};
    pack_le(w, 0, 1, OPCODE_GDMA_GM2UB);
    pack_le(w, 1, 2, instr_idx_++);
    pack_le(w, 3, 5, (uint64_t)gm_addr);
    pack_le(w, 8, 3, ub_addr);
    pack_le(w, 11, 2, tensor_m);
    // ... 其余字段
    for (int i = 0; i < 8; i++) lpu_vstore(i, w[i]);
    lpu_vdispatch();
}
```

**验收标准：**能 \#include "lpu\_isa\.h" 并编译，字段顺序和 task\_gen\.py 的 gdma\_gm2ub 方法对齐。

## 第 3 步 — Host\-mock 指令 trace 对比

**目标：**不需要 rv32 工具链，先验证 lpu\_isa\.h 的字段编码是否正确。

写一个 host 测试（rv32sdk/tests/host\_mock/test\_matmul\_trace\.cpp）：

1. 定义 mock dispatch callback，把每次 lpu\_vdispatch 的 VREG 快照追加到 vector\<array\<uint32\_t,8\>\>

2. 用 host 编译器编译 ops/matmul\.cpp，link mock callback

3. 运行 op\_matmul\(params\)，得到 dispatch trace

4. 用 pymodel gen\_taskbin\.py 对同一参数生成 golden 指令流（文本格式）

5. 逐条 diff

这一步完全是 x86 编译，没有 rv32 工具链依赖，可以快速迭代。

**验收标准：**小矩阵 matmul（例如 M=16, N=8, K=8）的所有 dispatch 指令（gdma/ldma/mxu/set\_flag/wait\_flag）和 pymodel golden 逐字节一致。

## 第 4 步 — C\+\+ 算子库 rv32sdk/ops/

对照 pymodel ops/\*\.py 翻译每个算子，去掉 Python golden 那一侧（isa\.X\(\) 调用），只保留 lpu\_isa\.h 调用路径。

实现顺序（和现有 pymodel 测试覆盖顺序对齐）：

1. matmul\.cpp — 最核心，验证 GDMA/LDMA/MXU/ARU 全链路

2. rmsnorm\.cpp

3. softmax\.cpp

4. attention\.cpp（依赖前三个）

5. ffn\.cpp

每个算子对应一个 host\-mock 测试（第 3 步的扩展），保证在进入 rv32 之前编码就是对的。

## 第 5 步 — Bare\-metal 启动 \+ RV32 ELF 编译

**rv32sdk/startup/crt0\.S**（极简，20 行以内）：

```asm
.section .text.start
.global _start
_start:
    la sp, __stack_top    # 初始化栈指针
    call main
    li a7, 93             # ecall exit
    ecall
```

**rv32sdk/startup/link\.ld**：

```text
MEMORY {
    MEM : ORIGIN = 0x00000000, LENGTH = 192K
}
SECTIONS {
    .text : { *(.text*) } > MEM
    .data : { *(.data*) } > MEM
    .bss  : { *(.bss*)  } > MEM
    . = ALIGN(4);
    __stack_top = ORIGIN(MEM) + LENGTH(MEM);
}
```

编译命令：

```bash
riscv64-unknown-elf-g++ -march=rv32im -mabi=ilp32 -nostdlib \
    -T link.ld startup/crt0.S ops/matmul.cpp -I include/ \
    -o matmul_test.elf
```

**验收标准：**readelf \-h 显示 Machine: RISC\-V, Class: ELF32。

## 第 6 步 — RV32 Function Model cmodel/src/rv32/

四个模块，从底到顶：

**rv32\_mem\.h/cpp** — 平坦内存 \+ ELF 加载：

- 内存：192KB uint8\_t mem\[0x30000\]，支持 load8/16/32 和 store8/16/32

- ELF 加载：使用 ELFIO，遍历 PT\_LOAD segments，memcpy 到对应偏移，返回 entry point 地址

**rv32\_cpu\.h/cpp** — RV32I\+M 解释器（约 1200 行）：

- 寄存器：uint32\_t xreg\[32\]，PC

- 主循环：fetch → decode（按 opcode/funct3/funct7 switch）→ 更新寄存器/内存/PC

- 需要实现的指令集：RV32I 的全部 47 条 \+ RV32M 的 8 条乘除法（地址计算需要）

- ecall with a7=93：设置 halted\_ = true

**xlpuv\_ext\.h/cpp** — xlpuv 处理：

- uint32\_t vreg\[8\]

- 识别 opcode=0x0B，按 funct3 分派 vstore/vload/vdispatch

- vdispatch 时调用 dispatch\_callback\_\(vreg\)

```cpp
using DispatchCallback = std::function<void(const uint32_t vreg[8])>;
```

**rv32\_top\.h/cpp** — 顶层组装：

```cpp
class RV32Top {
public:
    void load_elf(const std::string& path);
    void set_dispatch_callback(DispatchCallback cb);
    void run(size_t max_instrs = 100'000'000);
    const std::vector<std::array<uint32_t,8>>& dispatch_trace() const;
private:
    RV32Mem mem_;
    RV32Cpu cpu_;
    XlpuvExt xlpuv_;
    DispatchCallback dispatch_cb_;
    std::vector<std::array<uint32_t,8>> trace_;
};
```

CMake：

```cmake
# cmodel/src/rv32/CMakeLists.txt
add_library(rv32sim STATIC rv32_cpu.cpp rv32_mem.cpp xlpuv_ext.cpp rv32_top.cpp)
target_include_directories(rv32sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third_party/elfio)
```

在 cmodel/src/CMakeLists\.txt 里加 add\_subdirectory\(rv32\)。

**验收标准：**写一个 20 行的单元测试，加载第 5 步生成的 matmul\_test\.elf，以 trace 模式运行，dispatch\_trace 非空。

## 第 7 步 — RV32 \+ FuncTop Co\-Sim 测试

**cmodel/func\_tests/rv32/test\_rv32\_cosim\.cpp**：

```cpp
class RV32CoSimTest : public ::testing::Test {
    LPU::FuncTop func_top_;
    RV32Top      rv32_top_;
};

TEST_F(RV32CoSimTest, matmul_smoke) {
    // 1. 在 FuncTop 的 GM 里写入测试数据
    func_top_.write_gm(left_gm_addr, left_data.data(), left_data.size());
    func_top_.write_gm(right_gm_addr, right_data.data(), right_data.size());

    // 2. 设置 dispatch callback：把 VREG 转换为 ISA_t 并 enqueue
    rv32_top_.set_dispatch_callback([&](const uint32_t vreg[8]) {
        uint8_t raw[32];
        for (int i = 0; i < 8; i++) memcpy(raw + i*4, &vreg[i], 4);
        LPU::ISA_t instr = LPU::decode_instruction(raw);
        func_top_.enqueue_instruction(instr);
    });

    // 3. 加载算子 ELF，运行
    rv32_top_.load_elf("rv32sdk/build/matmul_test.elf");
    rv32_top_.run();

    // 4. 执行 FuncTop（消费 enqueue 的指令）
    func_top_.run_instruction_stream(/*sequential=*/false);

    // 5. 写 golden，比较
    func_top_.write_golden_gm(result_gm_addr, golden.data(), golden.size());
    EXPECT_TRUE(func_top_.check_gm_range(result_gm_addr, golden.size()));
}
```

因为 func\_tests/rv32/ 在 file\(GLOB\_RECURSE \.\.\.\) 扫描范围内，这个测试会自动加入 runFuncTest，不需要修改 CMakeLists\.txt。

**验收标准：**\./build/runFuncTest \-\-gtest\_filter=RV32CoSimTest\.\* 全部通过，结果和 pymodel golden 数值一致。

## 第 8 步 — Qwen3 算子 \+ vLLM 接入（后续）

在 co\-sim 通过后，把 pymodel/qwen3/ 的算子也用 C\+\+ 重写，走同一套 RV32 编译 \+ co\-sim 验证流程。vLLM 的 task\_gen\.py 可以逐步替换为"加载 rv32 ELF \+ RV32Top\.run\(\) \+ 收集 dispatch trace → 生成 taskbin bytes"。

# 七、硬件部分（后续）

硬件部分（RISC\-V 核选型、IS back\-pressure RTL 接口、Ibex co\-processor interface）等软件仿真路跑通后再讨论。推荐核心为 Ibex（lowRISC，RV32IMC），使用其官方文档化的 X\-IF co\-processor interface。

xlpuv\_copro\.sv 逻辑：

- 监听 Ibex X\-IF offload 接口，识别 opcode=0x0B

- lpu\_vstore：写 VREG\[rs1\] = rs2

- lpu\_vload：读 VREG\[rs1\] → rd

- lpu\_vdispatch：把 VREG\[7:0\] 拼成 256bit，assert 到现有 SU dispatch 总线

RISC\-V 核运行的程序从 LMB 或一块独立的指令 SRAM 加载（ELF 由 host 通过 AXI 写入），reset vector 跳转过去。现有 SU 单元测试框架（rtl/unit\_test/su\_uvm/）可以改造成 RISC\-V 程序驱动的测试，替代原来的静态 taskbin 注入。
