# LPU Tutorial

先按 [快速通关学习路线](LEARNING_ROADMAP.md) 学习和修复，再运行下面的端到端测试。

```bash
python3 -m unittest test.BlockTestMatmul.test_matmul_tile_twice
python3 -m unittest test.BlockTestMatmul.test_matmul_tile_once
python3 -m unittest test.BlockTestMatmul.test_matmul_transpose
python3 -m unittest test.BlockTestActivation.test_softmax
```
