# LPU Tutorial 快速通关学习路线

> 目标：把一个目前“能读、能编译、但测试全红”的 LPU Function Model，按可验证的小步骤修成能运行的 tutorial。  
> 方法：一次只修一层；每 15～45 分钟拿到一个绿色测试；不要同时改所有算子。  
> 当前日期：2026-07-19。

## 1. 先确定什么叫“做完”

这个仓库覆盖的是 OpenTPU/LPU 设计流程中的 **ISA 定义与 Python Function Model 验证**，不是整个芯片项目。

### 今晚最小完成线（P0）

README 中的 4 个核心测试全部通过：

```bash
python3 -m unittest test.BlockTestMatmul.test_matmul_tile_once -v
python3 -m unittest test.BlockTestMatmul.test_matmul_transpose -v
python3 -m unittest test.BlockTestMatmul.test_matmul_tile_twice -v
python3 -m unittest test.BlockTestActivation.test_softmax -v
```

这代表你已经打通：布局转换 → DMA → MXU/ARU → 算子编排 → Golden 对比。

### 当前仓库完整完成线（P1）

```bash
python3 -m unittest -v
python3 ref/attention.py
python3 ref/transformer.py
```

验收标准：8 个 unittest 全绿，Attention 和 RoPE 参考实现均能运行。

### 后续项目线（P2，不要今晚展开）

- 补完 `ref/ffn.py`。
- 用已验证算子串起 Attention/FFN。
- 定义指令编码、set/wait 依赖和调度。
- Cycle-Accurate Model、性能评估、RTL。

只要 P0 完成，就可以明确地说“当前 tutorial 主线已做完”。CAModel 和 RTL 是下一阶段项目，不是今晚的隐藏作业。

## 2. 你现在并不是从零开始

审阅全部源码、文档和配图后的基线如下：

| 项目 | 当前状态 | 含义 |
|---|---:|---|
| 全部 Python 文件语法编译 | ✅ | 代码骨架是完整可读的 |
| `ref/attention.py` | ✅ | 普通、分头、Flash Attention 结果一致 |
| `ref/transformer.py` | ❌ | RoPE 矩阵法在广播维度处失败 |
| `python3 -m unittest -v` | 0/8 | 1 个失败、7 个报错 |
| `ref/ffn.py` | 空 | 是后续扩展，不阻塞 P0 |
| `isa.py`、`ops/matmul.py` | 有本地修改 | 这是你已有的工作，继续保留，不要 reset |

当前失败不是因为你“不会”，而是仓库经历过接口演进，`ops` 仍在调用旧版 `ISA` API，同时有几处 layout 和测试 Golden 的确定性错误。正确策略是恢复每层契约，而不是在最高层反复猜。

## 3. 先建立一张脑内地图

### 软件层次

```text
test.py                         PyTorch Golden + 最终验收
   ↓
ops/matmul.py, activation.py    算子如何切块、如何排列指令
   ↓
isa.py                          GDMA / LDMA / MXU / ARU 指令语义
   ↓
semantic.py                     Broadcast / Binary / Unary / Reduce 数学语义
   ↓
utils.py + common.py            Layout、常量、比较和辅助函数
```

出现错误时从下往上查。底层 shape 错误不修，高层算子越改越乱。

### 硬件数据流

```text
GM
├── A: K1MK0 ──GDMA/LDMA──> LMB: M1K1M0K0 ┐
├── B: K1NK0 ──GDMA/LDMA──> RMB: N1K1N0K0 ├──> MXU
└── bias: N ──────GDMA────> PMB: N1N0      ┘      │
                                                   v
                                             PSB: M1N1M0N0
                                                   │
                                                   v
                                                  ARU
                                                   │
                                              UB 或 GM
```

模块职责只记一句话即可：

- GDMA/LDMA：搬数据，同时做 layout 变换。
- MXU：块矩阵乘、加 bias、累加 partial sum。
- ARU：Binary → Unary → Reduce → 写回。
- IS：未来用 set/wait 管理模块之间的数据依赖；当前 Python FModel 暂不模拟周期。

## 4. 必须掌握的 shape 语言

当前代码以 `M0=N0=K0=8` 为准。项目简介写了 `M0=16`，这是文档与代码的已知差异；先让 Function Model 按代码常量通过，之后再统一规格。

取一个固定例子：

```text
M=10, N=9, K=16
M1=ceil(10/8)=2
N1=ceil(9/8)=2
K1=ceil(16/8)=2
```

| 名称 | shape | 逻辑含义 |
|---|---|---|
| `A_mk` | `[M,K]` | 普通左矩阵 |
| `A_k1mk0` | `[K1,M,K0]` | GM/UB 左矩阵 |
| `A_m1k1m0k0` | `[M1,K1,M0,K0]` | LMB 左矩阵 |
| `B_nk` | `[N,K]` | 普通右矩阵；计算使用 `B.T` |
| `B_k1nk0` | `[K1,N,K0]` | GM/UB 右矩阵 |
| `B_n1k1n0k0` | `[N1,K1,N0,K0]` | RMB 右矩阵 |
| `bias_n1n0` | `[N1,N0]` | PMB bias |
| `C_m1n1m0n0` | `[M1,N1,M0,N0]` | PSB 输出 |
| `C_mn` | `[M,N]` | 最终逻辑输出 |

你应该能口头解释这条公式：

```text
C[m,n] = bias[n] + sum(A[m,k] * B[n,k], k=0..K-1)
```

### 最重要的 pack/unpack 逻辑

```text
MK [M,K]
  → 把 K 补齐到 K1*K0
  → reshape [M,K1,K0]
  → permute [K1,M,K0]
```

反向操作按相反顺序进行，并在最后裁掉 padding。任何 `permute(1,0,2)` 之前，输入必须真的是 3 维；二维 tensor 不能直接做三维 permute。

## 5. 通关规则：减少无效调试

1. **每次只做下面的一张任务卡。** 当前关没绿，不进入下一关。
2. **先固定随机性。** 在测试入口加入 `np.random.seed(0)` 和 `torch.manual_seed(0)`。
3. **先用 CPU 浮点路径。** 暂时统一 float32/fp16；P0 不碰 int8、bf16 和性能优化。
4. **所有 ISA 调用使用命名参数。** 这个仓库最常见的问题就是位置参数错位。
5. **变量名就是 shape。** 例如 `right_k1nk0`，每次 reshape 后立刻写 shape assert。
6. **先固定 tile，再恢复随机 tile。** 随机切块不应参与第一次正确性调试。
7. **每关 Red → Green → 记录。** 不要等 8 个测试一起绿才确认进展。
8. **卡住 25 分钟就停。** 把“关卡编号 + 命令 + 完整 traceback + 你认为的输入/输出 shape”发给我。

## 6. 今晚 P0 冲刺路线

建议总耗时 4～8 个专注小时。每完成一关就在标题前打勾；如果时间不足，至少完成到关卡 6，你会得到第一个端到端绿色算子。

### [x] 关卡 0：让测试成为可信的仪表盘（已完成）

目标：先消除测试自身的错误和随机漂移。

修改点：

- 在 `test.py` 固定 NumPy 和 PyTorch seed。
- LayerNorm Golden 改用 `torch.nn.functional.layer_norm(left, (left.shape[-1],))`。
- RMSNorm Golden 改用 `torch.nn.functional.rms_norm(left, (left.shape[-1],))`。
- SiLU Golden 改用 `torch.nn.functional.silu(left)`。
- 暂时不要放宽误差阈值来“换绿色”。

验收：这三个测试不再死在 Golden API，而是能进入你的实现。

```bash
python3 -m unittest test.BlockTestActivation.test_layernorm -v
python3 -m unittest test.BlockTestActivation.test_rmsnorm -v
python3 -m unittest test.BlockTestActivation.test_silu -v
```

绿色奖励：你已经把“测试坏了”和“实现坏了”分开，这是工程调试的第一步。

### [x] 关卡 1：打通 layout 往返（已完成）

目标：普通矩阵 pack 后再 unpack，必须逐元素回到原值，尤其是非 8 倍数尺寸。

先为以下函数写小单测，建议新建 `test_primitives.py`：

- `mk_to_k1mk0`
- `k1mk0_to_mk`
- `k1mk0_to_m1k1m0k0`
- `m1k1m0k0_to_k1mk0`

固定用例：`M=10, K=17`，不要先用随机数。

已知检查点：

- `utils.py` 当前把 `[:, M]` 写成了单点索引，应思考这里为何需要 `[:, :M]`。
- padding 后应该 reshape 新的 padded tensor，而不是原 tensor。
- `[M1,K1,M0,K0]` 的第 0 维是 M1、第 1 维才是 K1；反解时不要读反。
- 新建 tensor 时保持原输入的 `dtype` 和 `device`。

验收性质：

```python
x = torch.arange(10 * 17).reshape(10, 17)
packed = mk_to_k1mk0(x)
restored = k1mk0_to_mk(packed, 17)
assert torch.equal(restored, x)
```

再验证四维布局的 round-trip：

```text
K1MK0 → M1K1M0K0 → K1MK0
```

```bash
python3 -m unittest test_primitives.LayoutTest -v
```

### [x] 关卡 2：让 Semantic 层独立全绿（已完成）

目标：不经过 ISA，直接证明 `Broadcast/Binary/Unary/Reduce` 正确。

#### 2A. Broadcast

覆盖三种输入：

- scalar 广播到 `[M,N]`
- 长度 N 的向量沿 M 广播
- 长度 M 的向量沿 N 广播

用 `M=10,N=9` 验证逻辑区和 padding 区。保持 dtype/device。

特别检查 `isa.py` 对 Broadcast 的调用顺序。其语义应该是：

```python
Broadcast(arb_in, slice_m, slice_n, br_m, br_n)
```

#### 2B. Binary/Unary

每种操作只用一个 `2x3` tensor 验证。第二输入存在时，Binary flag 应当**恰好一个**为真；非法组合应明确报错，不要静默返回 `None`。

#### 2C. Reduce

输入物理布局是 `[M1,N1,M0,N0]`，逻辑矩阵却是 `[M,N]`。推荐逻辑：

```text
[M1,N1,M0,N0]
  → permute [M1,M0,N1,N0]
  → reshape [M1*M0,N1*N0]
  → 裁剪 [:M,:N]
  → 在逻辑 M 或 N 维 reduce
```

这一步能一次解决两个问题：同时 reduce 外层与内层轴，并排除 padding 对 max/min/mean/exp 的污染。

```bash
python3 -m unittest test_primitives.SemanticTest -v
```

### [x] 关卡 3：逐条打通 DMA（已完成）

目标：只测试搬运与 layout，不让 MXU/ARU 干扰定位。

按顺序验证：

1. `gdma_mov2ub`
2. `gdma_mov2lmb`
3. `gdma_mov2rmb`
4. `gdma_mov2pmb`
5. `ldma_mov2lmb`
6. `ldma_mov2rmb`
7. `ldma_mov2rmb_transpose`

每个测试检查三件事：shape、dtype、几个手算位置。再加一个超出逻辑边界的 case，确认 padding 为 0 且不越界。

实现纪律：

- `start_*` 是起始坐标，`slice_*` 是本次长度，`tensor_*` 是完整逻辑尺寸，三者不要互换。
- K 与 K1 是不同单位；变量名中必须写清楚。
- 普通 LDMA 和 transpose LDMA 应保持输入 dtype。
- 用命名参数调用，避免当前 ops 中已有的位置参数错位。

```bash
python3 -m unittest test_primitives.ISADMATest -v
```

### [x] 关卡 4：单独打通一个 MXU tile（已完成）

> 如果还不能稳定读出 `tensor[a:b, i, :]` 的 shape，请先完成
> [Tensor 基础补给站](TENSOR_BASICS.md)。这不是绕路；MXU 后续几乎每一行都依赖索引、reshape 和 permute。

目标：暂时绕过 `ops`，直接喂给 `ISA.mxu_matmul` 合法的 LMB/RMB/PMB/PSB。

第一组：`M=5,N=7,K=8`，一次 K 计算，带 bias。  
第二组：`M=5,N=7,K=16`，分成两段 K，验证 `psum_en` 累加。

核心逻辑：

```text
for m1, n1:
    temp[M0,N0] = 0
    for k1:
        temp += LMB[m1,k1] @ RMB[n1,k1].T
    第一个 K tile 加 bias
    后续 K tile 累加到 PSB
```

当前 `mxu_matmul` 默认 `dtype='int8'`，但测试输入是浮点；P0 必须显式选择浮点路径，否则小数会被截断为整数。

```bash
python3 -m unittest test_primitives.ISAMXUTest -v
```

### [x] 关卡 4B：打通最小 ARU 写回（已完成）

目标：先只实现 matmul 所需的 PSB passthrough，不在这里同时解决全部激活功能。

固定契约：从 PSB 读 `[M1,N1,M0,N0]`，不做 Binary/Unary/Reduce，按固定返回 list 契约写到 UB 或 GM；调用方显式取出 list 中的 tensor。使用关卡 1 已通过的 layout 转换还原 `[M,N]`。

```bash
python3 -m unittest test_primitives.ISAARUPassthroughTest -v
```

这一小关必须在 `matmul_tile_once` 之前通过；完整的 scalar、broadcast、reduce 契约留到关卡 8。

### [x] 关卡 5：第一个 Boss——`matmul_tile_once`（已完成）

目标：拿到第一个端到端绿色测试。

正确流水线：

```text
left[M,K] / right[N,K] / bias[N]
  → pack 到 K1MK0 / K1NK0 / N1N0
  → GDMA/LDMA 到 LMB/RMB/PMB
  → MXU 写 PSB
  → ARU/布局转换写回
  → unpack 为 [M,N]
  → 对比 left @ right.T + bias
```

检查清单：

- 二维矩阵必须先 `reshape` 成三维，才能 `permute`。
- 暂时固定 tile；不要调用随机 tiling 搜索。
- K 方向循环统一使用 K1 单位。
- `bias_en` 只在第一个 K tile 为真。
- `psum_en` 只在后续 K tile 为真。
- 最终输出裁剪到 `[:M, :N]`，不是 `[:M, :M]`。
- 更新成当前 `ISA` 函数签名，全部用命名参数。
- 先复用关卡 4B 已通过的 ARU passthrough，不要一边修 matmul 一边设计激活指令。

```bash
python3 -m unittest test.BlockTestMatmul.test_matmul_tile_once -v
```

通过后再用 `M=10,N=9,K=16` 手动跑一次并打印每层 shape。拿到这个绿色，说明主干已真正贯通。

### [ ] 关卡 6：`matmul_transpose`（30～60 分钟）

目标：支持 Attention 中的 `Q @ K.T`。

测试传入的右矩阵是 `[K,N]`。先明确逻辑等价关系：

```text
right_kn[K,N].T = right_nk[N,K]
```

再把它组织成 LDMA transpose 所期望的 `[N,K1,K0]`，最终写成 RMB `[N1,K1,N0,K0]`。不要对二维 `right_kn` 直接使用三维 permute。

你已经在 `isa.py` 和 `ops/matmul.py` 做了 transpose 相关修改；保留这些工作，只需用关卡 1～4 的 shape 契约逐项对齐调用参数。

```bash
python3 -m unittest test.BlockTestMatmul.test_matmul_transpose -v
```

### [ ] 关卡 7：`matmul_tile_twice`（45～90 分钟）

目标：在关卡 5 的正确实现上增加 L2→L1→L0 两级切块，不重新发明矩阵乘。

推荐做法：

1. 复制关卡 5 已验证的数据流思路。
2. 先只让 M/N 跨 tile，K 不跨 tile。
3. 再让 K 跨 tile，验证 partial sum。
4. 最后才恢复随机 tile。

重点排查：

- 右矩阵调用 `gdma_mov2ub` 时，`tensor_n/tensor_k1` 是否传反。
- L1/L0 的 `start_k1` 和 `size_k1` 是否始终以 K1 为单位。
- PMB 是从完整 bias 按全局 N offset 取数，不能把局部长度当完整长度。
- PSB 索引需要由当前 M tile 和 N tile 共同决定。
- 只有最后一个 K tile 才输出；第一个 K tile 才加 bias。

```bash
python3 -m unittest test.BlockTestMatmul.test_matmul_tile_twice -v
```

### [ ] 关卡 8：统一 ARU 契约（30～60 分钟）

目标：让所有 activation 都依赖同一套、可预测的 ARU 行为。

为了最快完成，建议保留当前设计：ARU 始终返回 list，顺序固定为：

```text
1. UB 或 GM 写回结果（若启用）
2. ARB 写回结果（若启用）
```

调用方必须显式解包，不能有时把 list 当 tensor、有时当 tuple。

至少写 5 个小测试：

- passthrough
- scalar add
- M/N 向量广播
- unary exp/reciprocal
- reduce 并写入 ARB

同时处理：

- ARB-only 输入是否合法。
- 第二输入存在时 Binary flag 恰好一个。
- `ub_wr_en/gm_wr_en/arb_wr_en` 的返回顺序。
- `ub_layout=1` 使用已经验证过的 layout converter。

```bash
python3 -m unittest test_primitives.ISAARUTest -v
```

### [ ] 关卡 9：第二个 Boss——稳定 Softmax（30～60 分钟）

目标：完成 README 的第 4 个核心测试。

只记住这四步，不要边写边猜 flags：

```text
row_max = reduce_max(x, N)
e       = exp(x - broadcast(row_max))
row_sum = reduce_sum(e, N)
y       = e / broadcast(row_sum)
```

当前代码的检查点：

- 不要在变量赋值前读取 `exp_m1n1m0n0_ub`。
- 第二步应该读取原 x，写出 e；第三步对 e 求和。
- 最后一步是 `div`，不是再次 `sub + exp + reduce`。
- `return` 必须移出 M tile 循环，否则只处理第一个 tile。
- padding 不得进入 max 或 sum。

```bash
python3 -m unittest test.BlockTestActivation.test_softmax -v
```

### [ ] P0 总验收：README 主线完成

```bash
python3 -m unittest test.BlockTestMatmul.test_matmul_tile_once -v
python3 -m unittest test.BlockTestMatmul.test_matmul_transpose -v
python3 -m unittest test.BlockTestMatmul.test_matmul_tile_twice -v
python3 -m unittest test.BlockTestActivation.test_softmax -v
```

四条全绿后，给自己打一个明确里程碑：**LPU Function Model 的核心矩阵与 Softmax 数据流已贯通。**

## 7. P1：把当前仓库全部点亮

P0 之后继续。这里每个算子都建立在同一个 ARU 契约上，会明显快很多。

### [ ] 关卡 10：Sigmoid（20～40 分钟）

```text
sigmoid(x) = reciprocal(1 + exp(-x))
```

检查当前实现中的 `neg_en`、未定义变量、scalar add、reciprocal 和写回目标。

```bash
python3 -m unittest test.BlockTestActivation.test_sigmoid -v
```

### [ ] 关卡 11：SiLU（15～30 分钟）

```text
silu(x) = x * sigmoid(x)
```

复用已经通过的 Sigmoid 指令链；最后必须启用 mul，并同时保留原 x 与 sigmoid(x)。

```bash
python3 -m unittest test.BlockTestActivation.test_silu -v
```

### [ ] 关卡 12：RMSNorm（30～45 分钟）

```text
mean_sq = mean(x*x, N)
inv_rms = reciprocal(sqrt(mean_sq + eps))
y       = x * broadcast(inv_rms)
```

先约定 `eps`，然后用 scalar add 表达。注意最后是乘逆值，不是把原值直接除以一个缺少 eps 的临时量。

```bash
python3 -m unittest test.BlockTestActivation.test_rmsnorm -v
```

### [ ] 关卡 13：LayerNorm（30～60 分钟）

```text
mean     = mean(x, N)
centered = x - broadcast(mean)
var      = mean(centered*centered, N)
inv_std  = reciprocal(sqrt(var + eps))
y        = centered * broadcast(inv_std)
```

当前代码把第一次 reduce mode 写成了 max，而且最后遗漏 centered；逐步保留中间值，不要压成一个超长 ARU 调用。

```bash
python3 -m unittest test.BlockTestActivation.test_layernorm -v
```

### [ ] 关卡 14：全套稳定性验收（20～40 分钟）

```bash
python3 -m unittest -v
python3 ref/attention.py
```

全绿后再做 10 次压力运行，确认随机尺寸不会偶发失败：

```bash
for i in $(seq 1 10); do
  python3 -m unittest -q || exit 1
done
```

通过标准：不是“平均误差看起来还行”，而是 shape 正确、无 NaN/Inf、每次都通过。

## 8. P2：从算子走向 Transformer

### [ ] 关卡 15：把 Reduce 实验脚本变成可信测试

`ref/reduce.py` 当前主要用于打印观察，缺少断言；padding 为 0 时也会让全负数 max 得到错误结果。先改成可导入、无顶层副作用的函数，再覆盖：N=8、N=13、全负数、同时 reduce M/N 和尾块不足 `P_ARU`。

```bash
python3 -m ref.reduce
```

这不是 P0 阻塞项；它用于在进入周期级 ARU 设计前留下可信 Golden。

### [ ] 关卡 16：修 RoPE 参考实现

`ref/transformer.py` 的逐元素 RoPE 已可作为 Golden；矩阵法目前在 `torch.matmul(x, rope_matrix)` 处批次维广播不匹配。

思路：每个 sequence 位置使用自己的 `[dim,dim]` 旋转矩阵。可显式增加 batch/head 维，或用 `einsum` 写清：

```text
[batch,head,seq,dim] × [seq,dim,dim] → [batch,head,seq,dim]
```

```bash
python3 ref/transformer.py
```

验收：`torch.allclose(result1, result2)` 为 True。

### [ ] 关卡 17：补 FFN Golden

`ref/ffn.py` 当前为空。先只实现数学 Golden，不接 ISA：

```text
普通 FFN:  y = W2(activation(W1(x)))
SwiGLU:    y = W_down(silu(W_gate(x)) * W_up(x))
```

写固定 shape 和 assert，为未来算子串联留一个可信目标。

### [ ] 关卡 18：串起最小 Attention Block

按这个顺序复用已通过算子：

```text
Q = XWq
K = XWk
V = XWv
S = QK^T / sqrt(head_dim)
P = softmax(S + mask)
O = PV
Y = OWo
```

先单 head、无 mask，再多 head，再 causal mask。每增加一步都与 `ref/attention.py` 对比，不要一次写完整 Transformer。

## 9. 已知坑位地图

遇到错误先查这张表，可以省掉大量盲查时间。

| 位置 | 已知问题 | 应在哪一关处理 |
|---|---|---:|
| `test.py` LayerNorm/RMSNorm/SiLU Golden | PyTorch API 用法错误 | 0 |
| `utils.py` 四维 layout 往返 | 范围索引、padding tensor、M1/K1 读取错误 | 1 |
| `semantic.py` Broadcast | dtype/device 与调用参数顺序 | 2 |
| `semantic.py` Reduce | 只 reduce 外层轴且未屏蔽 padding | 2 |
| `isa.py` DMA | dtype 和 start/size/full-size 约定不一致 | 3 |
| `isa.py` MXU | 默认 int8 会截断浮点测试数据 | 4 |
| `ops/matmul.py` once/transpose | 二维 tensor 直接三维 permute | 5/6 |
| `ops/matmul.py` 两级 tiling | K 与 K1 单位、位置参数和 offset 混用 | 7 |
| `isa.py` ARU | Broadcast 调用顺序、输入组合、返回 list 契约 | 8 |
| `ops/activation.py` | 未定义变量、错误 flags、循环内提前 return | 9～13 |
| `ref/reduce.py` | padding、跨 M1 累积、尾块与缺少断言 | 15 |
| `ref/transformer.py` | RoPE 矩阵批次广播失败 | 16 |
| `common.py` vs 项目简介 | 代码 M0=8，文档 M0=16 | P1 后统一 |

还有两个原则性坑：

- `compare()` 当前只看平均相对误差且默认阈值较松，可能掩盖局部大错。基础单测优先使用 `torch.testing.assert_close`。
- `from ... import *` 让依赖来源模糊。先完成 P0，再做 import 清理；不要把重构和正确性修复混在一起。

## 10. 文档该怎么读，不必从头背

按下面顺序阅读 `LPU项目简介/LPU项目简介.md`：

1. 需求与算子范围：15～45 行。
2. 模块职责：47～97 行。
3. 一次 matmul 的核内数据流：113～137 行。
4. Data Path 和 Layout：139～178 行。
5. set/wait 与执行依赖：180～223 行。
6. 项目全流程：227～259 行。

三张图分别对应整体模块、L2/L1/L0 内存层次、GDMA→LDMA→MXU→ARU 依赖时序。看图时只回答两个问题：数据现在在哪个 Buffer？下一条指令由哪个模块执行？

文档里 PMB/PB、mov2pmb/mov2pb、GM/DDR 有命名混用，它们不是额外的新概念。当前 tutorial 的重点止于 Python Function Model；set/wait、CAModel 和 RTL 暂不阻塞测试。

## 11. 每完成一关这样记录

把下面模板复制到本文件末尾或自己的学习日志：

```markdown
### 关卡 X：名称

- 开始时间：
- 我先写下的输入 shape：
- 我预期的输出 shape：
- 第一个错误：
- 根因：
- 我修改了：
- 验收命令：
- 结果：PASS / FAIL
- 我现在能解释的一句话：
- 下一关：
```

建议里程碑：

- [ ] 铜牌：Layout + Semantic 小测试全绿。
- [ ] 银牌：第一个 `matmul_tile_once` 全绿。
- [ ] 金牌：README 四项 P0 全绿。
- [ ] 全成就：8 个 unittest + Attention + RoPE 全绿。

## 12. 现在只做第一件事

不要继续通读全部源码。先完成 **关卡 0**，然后运行：

```bash
python3 -m unittest test.BlockTestActivation.test_layernorm -v
```

把新的 traceback 和你修改后的三行 Golden 发给我。下一轮只处理 traceback 暴露出来的第一层问题，我会继续陪你逐关推进。
