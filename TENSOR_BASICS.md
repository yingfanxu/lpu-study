# Tensor 基础补给站

> 目标：不是背 PyTorch API，而是看到一行 Tensor 代码时，能写出每个维度的含义和输出 shape。  
> 用时：约 60～90 分钟。一次只做一节，不要先把全文看完。  
> 完成后返回 `LEARNING_ROADMAP.md` 的关卡 4。

## 学习方法

每个例子都按固定循环：

1. 先看输入 `shape`。
2. 给每个维度写名字。
3. 不运行代码，先猜输出 `shape`。
4. 运行验证。
5. 如果猜错，只解释错在哪个维度。

需要真正掌握的只有五条规则：

```text
1. shape 中每个数字代表一个维度的长度。
2. 索引中每个逗号进入下一个维度。
3. 整数索引会删除这一维，切片会保留这一维。
4. reshape 不改变元素总数。
5. permute 只调整维度顺序，不改变数据总数。
```

进度：

- [ ] A. shape 与维度
- [ ] B. 整数索引与切片
- [ ] C. reshape
- [ ] D. permute
- [ ] E. Broadcast 与 Reduce
- [ ] F. LPU 分块 layout

## A. shape 与维度

运行：

```bash
python3 - <<'PY'
import torch

a = torch.tensor(5)
b = torch.arange(4)
c = torch.arange(6).reshape(2, 3)
d = torch.zeros(2, 3, 4)

print("a：", a.shape, "ndim =", a.ndim)
print("b：", b.shape, "ndim =", b.ndim)
print("c：", c.shape, "ndim =", c.ndim)
print("d：", d.shape, "ndim =", d.ndim)
PY
```

你需要能读成：

```text
a：标量，0维
b：长度4的一维向量
c：2行3列的二维矩阵
d：第一维2、第二维3、第三维4的三维 Tensor
```

完成标准：能解释 `shape=[2,3,4]` 有三个维度，共 `2×3×4=24` 个元素。

## B. 整数索引与切片

先记符号：

```text
:     这一维全部
:2    下标0、1
1:3   下标1、2
2:    从下标2到结尾
2     只取下标2，并删除这一维
```

运行前先猜每一行的 shape：

```bash
python3 - <<'PY'
import torch

x = torch.zeros(3, 4, 8)
print("x                  ", x.shape)
print("x[1]               ", x[1].shape)
print("x[1, 2]            ", x[1, 2].shape)
print("x[1, 2, :]         ", x[1, 2, :].shape)
print("x[1:3, 1:3, :]     ", x[1:3, 1:3, :].shape)
print("x[:, 2, :]         ", x[:, 2, :].shape)
print("x[:, 2:3, :]       ", x[:, 2:3, :].shape)
PY
```

答案：

```text
x                   [3,4,8]
x[1]                [4,8]
x[1,2]              [8]
x[1,2,:]            [8]
x[1:3,1:3,:]        [2,2,8]
x[:,2,:]            [3,8]
x[:,2:3,:]          [3,1,8]
```

项目里的读法：

```text
x.shape = [K1,M,K0]
x[1:3,1:3,:]
  K1取2个，M取2个，K0全部取出
  输出 [2,2,K0]
```

## C. reshape

`reshape` 只改变如何分组，元素总数必须相同。

```bash
python3 - <<'PY'
import torch

x = torch.arange(24)
a = x.reshape(3, 8)
b = x.reshape(2, 3, 4)
c = x.reshape(1, 2, 3, 4)

print(x.shape, "元素数 =", x.numel())
print(a.shape, "元素数 =", a.numel())
print(b.shape, "元素数 =", b.numel())
print(c.shape, "元素数 =", c.numel())
PY
```

这些 shape 都合法，因为：

```text
24 = 3×8 = 2×3×4 = 1×2×3×4
```

`24` 个元素不能直接 reshape 成 `[4,8]`，因为目标需要 `32` 个元素。需要先创建 padding。

## D. permute

`permute` 调整维度顺序。

```bash
python3 - <<'PY'
import torch

x = torch.zeros(3, 4, 8)
y = x.permute(1, 0, 2)

print("x [K1,M,K0]：", x.shape)
print("y [M,K1,K0]：", y.shape)
PY
```

理解方式：

```text
x 原维度编号：0=K1，1=M，2=K0
permute(1,0,2)：新顺序使用旧的 1、0、2
所以 [K1,M,K0] → [M,K1,K0]
```

元素对应关系：

```python
y[m, k1, k0] == x[k1, m, k0]
```

## E. Broadcast 与 Reduce

逻辑矩阵：

```python
x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
])
```

Broadcast：

```text
[10,20,30] 沿 M 广播
→ [[10,20,30],
   [10,20,30]]
```

Reduce：

```text
沿 N sum：每行压成一个数 → [6,15]
沿 M sum：每列压成一个数 → [5,7,9]
```

PyTorch 中：

```python
torch.sum(x, dim=1)  # 沿N，结果shape=[M]
torch.sum(x, dim=0)  # 沿M，结果shape=[N]
```

## F. LPU 分块 layout

块大小：

```text
M0=N0=K0=8
```

普通矩阵：

```text
[M,K] = [4,24]
```

K 被拆成：

```text
K1=24/8=3
[M,K] → [K1,M,K0] = [3,4,8]
```

最重要的元素对应关系：

```python
gm[k1, m, k0] == logical[m, k1 * 8 + k0]
```

对于 `[M1,N1,M0,N0]`：

```python
physical[m1, n1, m0, n0] == logical[
    m1 * 8 + m0,
    n1 * 8 + n0,
]
```

反向计算：

```text
m1 = m // 8    m0 = m % 8
n1 = n // 8    n0 = n % 8
```

## 毕业检查

给定：

```python
x.shape = [4, 5, 8]
```

不运行代码，写出：

```text
1. x[1].shape
2. x[1:3].shape
3. x[:,2,:].shape
4. x[:,2:3,:].shape
5. x[1:3,2:5,:].shape
6. x.permute(1,0,2).shape
```

全部答对后再回到 MXU。语法忘记时可以随时查这份文件；工程能力不是“不查资料”，而是能根据规则快速确认结果。
