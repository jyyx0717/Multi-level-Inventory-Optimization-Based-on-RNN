# 基于循环神经网络的多级库存优化项目

## 📖 项目简介

本项目是一个基于 **可微仿真** 与 **递归神经网络（RNN）** 理念的多级库存优化项目。

该项目通过构建供应链的“数字孪生”模型，利用自动微分技术计算总成本对库存水平的梯度，从而通过自适应矩估计 (Adam) 直接求解复杂约束下的最优库存策略。

项目将数据预处理、优化求解、仿真评估与闭环调整的完整工作流整合于一个 **Jupyter Notebook** 中，便于交互式调试与分析；而核心的微分仿真与优化算法封装于独立模块中以保证性能。

## 🚀 核心功能

*   **全链路建模**：支持从原材料（RM）到成品（FG），从工厂（IDC）到区域仓（RDC）的多级网络结构。
*   **复杂约束处理**：内置BOM（物料清单）逻辑，处理生产齐套率（木桶效应）；支持最小起订量（MOQ）、补货周期和提前期约束。
*   **交互式工作流**：在一个 Notebook 中完成从数据加载到策略生成的全过程。
*   **闭环优化流**：
    *   **Solve**: 寻找数学最优。
    *   **Evaluate**: 随机模拟验证服务水平。
    *   **Adjust**: 动态调整惩罚系数，平衡成本与服务水平。

## 📂 项目结构

```text
MEIO_Project/
├── data/                        # 数据存放目录 (CSV源文件及生成的PKL模型)
│   ├── 仓库主数据.csv
│   ├── 产品主数据.csv
│   ├── 生产物料清单.csv
│   ├── 销量分布.csv
│   ├── penalty_factor.csv       # (自动生成/更新) 缺货惩罚系数
│   └── ...
├── rnnisa/                      # 核心算法包
│   └── model/
│       ├── simulation_lead_real.py  # [核心] 仿真引擎与自动微分逻辑 (BPTT)
│       └── simu_opt.py              # [核心] 自适应矩估计优化器 (Adam) 
├── results/ 
├── main.ipynb                   # [主程序] 集成工作流 (预处理 -> 优化 -> 评估 -> 调整)
└── README.md                    # 项目说明文档
```

## 🛠️ 环境依赖

*   Python 3.8+
*   Jupyter Notebook / Lab
*   NumPy
*   Pandas
*   NetworkX
*   SciPy
*   Matplotlib

安装命令：
```bash
pip install numpy pandas networkx scipy matplotlib jupyter
```

## ⚡ 快速开始

所有的业务流程操作均在 `main.ipynb` 中完成。

### 第一步：准备数据
确保 `data/` 目录下包含以下基础 CSV 文件（需符合指定格式）：
*   `仓库主数据.csv`, `产品主数据.csv`, `生产物料清单.csv`
*   `IDC-CDC补货参数.csv`, `CDC-RDC补货参数.csv`
*   `销量分布.csv` (包含 Location, SKU, Mean, Std)
*   `penalty_factor.csv` 惩罚系数文件

### 第二步：运行 Notebook (`main.ipynb`)

启动 Jupyter Notebook 并打开 `main.ipynb`，按顺序执行其中的 Cell 模块：

1.  **数据预处理 (Preprocessing)**
    *   执行对应代码块。
    *   **功能**：读取 `data/` 下的 CSV，基于反向回溯逻辑构建供应链网络图。
    *   **输出**：`data/supply_chain_network.pkl`

2.  **核心优化 (Optimization / Solve)**
    *   执行优化代码块（调用 `rnnisa.model` 中的引擎）。
    *   **功能**：加载网络模型，从零库存或指定初始值开始，使用 Adam 算法寻找最低成本的库存策略。
    *   **输出**：`result_...csv` 推荐的库存策略（Base Stock Level），`cost_curves.png` 成本收敛曲线图。

3.  **服务水平评估 (Evaluation)**
    *   执行评估代码块。
    *   **功能**：使用蒙特卡洛模拟（Monte Carlo）在随机需求下测试生成的策略，计算 CSL 和 Fill Rate。
    *   **输出**：`data/service_level_summary.csv` (各节点服务水平报告)。

4.  **动态调整闭环 (Adjustment)**
    *   执行调整代码块。
    *   **功能**：如果评估显示服务水平未达标，自动调整 `penalty_factor.csv` 中的惩罚系数。
    *   **后续**：调整完成后，可重新运行 **核心优化** 代码块，进行新一轮迭代。

## ⚙️ 关键参数说明

这些参数通常在 `main.ipynb` 的配置 Cell 中定义：

*   **优化参数**:
    *   `MAX_EPOCHS`: 最大迭代次数（默认 100）。
    *   `LEARNING_RATE`: 学习率（默认 1e-3）。
    *   `TARGET_COST`: 目标成本，达到即早停。
    *   `TOLERANCE`: 优化解的收敛容差。
    *   `REP_NUM`: 每次梯度计算的样本采样数（用于平滑随机梯度）。
*   **调整逻辑**:
    *   `MIN/MAX_PENALTY_LIMIT`: 惩罚系数的硬性上下限。
    *   调整规则：服务水平缺口越大，惩罚系数放大的倍数越高（如 x1.5, x2.0, x3.0）。

## 📊 结果解读

1.  **策略文件 (`result_*.csv`)**:
    *   `Location` / `SKU`: 标识库存单元。
    *   `BaseStockLevel`: 建议的**库存水平**。业务含义：当 `库存位置 (IP) < BaseStockLevel` 时，触发补货/生产。

2.  **收敛曲线 (`cost_curves.png`)**:
    *   **总成本 (蓝线)**: 总体目标函数值，呈下降趋势。
    *   **持有成本 (红线)**: 随着优化进行，为了避免缺货，持有成本通常会先下降后上升，最终与缺货风险达成平衡。

3.  **评估报告 (`service_level_summary.csv`)**:
    *   `CSL`: 周期服务水平（不缺货天数占周期总天数的百分比）。
    *   `FR`: 需求满足率。
    *   通过此报告判断优化结果是否符合业务 KPI。

**参考文献:** Wang, Tan, and L. Jeff Hong. "Large-scale inventory optimization: A recurrent neural networks–inspired simulation approach." INFORMS Journal on Computing 35.1 (2023): 196-215.
