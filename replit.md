# Choice Engineering Competition - Experimental Platform

## Overview
This is a PHP-based web application for the Choice Engineering Competition, an academic competition that invites participants to devise mechanisms that engineer behavior. The application supports both human experiments and RL (Reinforcement Learning) agent experiments.

## Project Structure

### Main Components
- **experiment/main.php** - Main human experiment page
- **experiment/rl_api.php** - RL experiment API endpoint
- **experiment/rl_dashboard.html** - RL experiment control panel
- **experiment/scripts/backend.php** - Backend logic for human experiments
- **experiment/rl_agents/** - RL agent models
- **experiment/custom_sequences/** - Custom reward sequences

### Data Storage
- **results/** - Human experiment results (outside web root)
- **results_rl/** - RL experiment results (outside web root)
- **data/** and **data_unfiltered/** - Sample data files

## RL System (新增)

### 访问RL控制面板
- `/rl_dashboard.html` - 可视化控制面板，支持配置和运行RL实验

### RL API端点
```
GET /rl_api.php?action=<action>&model=<model>&schedule_type=<type>&schedule_name=<name>&trials=<n>
```

### API参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| action | init/run_trial/run_all/status/list_models/list_schedules | run_trial |
| model | RL模型名称 | simple_ql |
| schedule_type | STATIC/DYNAMIC | STATIC |
| schedule_name | 序列名称 | random_0 |
| trials | 试次数量 | 100 |

### 示例调用
```bash
# 完整运行实验
curl "/rl_api.php?action=run_all&model=simple_ql&schedule_type=STATIC&schedule_name=random_0&trials=100"

# 查看可用模型
curl "/rl_api.php?action=list_models"

# 查看可用序列
curl "/rl_api.php?action=list_schedules"
```

### 内置RL模型
- **simple_ql** - Q-Learning (ε-greedy)
- **random_agent** - 随机选择 (基线)
- **ucb_agent** - Upper Confidence Bound
- **neural_ql** - 神经网络增强Q-Learning (LSTM + Attention，从人类数据训练)
- **neural_ql_fast** - 快速规则模型 (45x faster, 基于人类行为规则)

## 模型评估系统

### 访问评估控制面板
- `/evaluate_dashboard.html` - 可视化评估界面

### 评估 API
```
GET /evaluate_api.php?models=<model>&runs=<n>&sample=<n>
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| models | fast/neural/all | all |
| runs | 随机模型运行次数 | 5 |
| sample | 样本数量 (0=全部) | 0 |

### 命令行评估
```bash
cd experiment/rl_agents/neural_ql
source /home/runner/workspace/.venv/bin/activate
python evaluate_all.py --model all --runs 1
```

### 当前模型准确率
| 模型 | 准确率 | 说明 |
|------|--------|------|
| Neural Network | 74.91% | LSTM + Attention |
| Fast Rules v2 | 64.30% | 8条行为规则 (优化后) |

### Fast Model v2 规则贡献率
| 规则 | 参数 | 贡献率 | 心理学原理 |
|------|------|--------|-----------|
| MOMENTUM | +20% | +4.95% | 行为惯性 |
| BASE_STAY | 66% | +2.68% | 基础一致性 |
| WIN_STAY | +15% | +0.80% | 热手效应 |
| PHASE | +10% (晚期) | +0.38% | 策略固化 |
| LEFT_BIAS | 62% | -0.13% | 左侧偏好 |
| STREAK_WIN | +5%/连胜 | -0.32% | 连胜信心 |
| RECENCY | 5% | -0.75% | 近因效应 |
| LOSE_SHIFT | 0% | - | (已禁用) |

### 消融实验
```bash
cd experiment/rl_agents/neural_ql
python ablation_study.py --runs 3 --sample 500
```

### 添加新RL模型
1. 在 `experiment/rl_agents/models/` 创建Python脚本
2. 接口: `python3 model.py <user_id> <last_action> <last_reward>`
3. 输出: 仅打印 "LEFT" 或 "RIGHT"
4. 详见 `experiment/rl_agents/README.md`

## 自定义序列

### 静态序列 (PHP)
创建文件 `experiment/custom_sequences/static/your_name.php`:
```php
<?php
$biased_rewards = [0, 1, 0, 1, ...];    // 100个元素
$unbiased_rewards = [1, 0, 1, 0, ...];  // 100个元素
?>
```

### 动态序列 (Python)
创建文件 `experiment/custom_sequences/dynamic/your_name.py`:
```python
import sys, json
bias_rewards = json.loads(sys.argv[1])
unbias_rewards = json.loads(sys.argv[2])
choices = json.loads(sys.argv[3])
print("1, 0")  # biased_reward, unbiased_reward
```

详见 `experiment/custom_sequences/README.md`

## Human Experiment

### 实验流程
1. 用户访问 `/main.php`
2. 选择左/右按钮 (100次试次)
3. 根据奖励分配方案获得反馈
4. 结果保存到CSV

### 配置
编辑 `experiment/main.php`:
```php
$_SESSION['schedule_type'] = $TYPE_STATIC; // or $TYPE_DYNAMIC
$_SESSION['schedule_name'] = "random_0";
```

## 运行环境

### 开发
```
php -S 0.0.0.0:5000 -t experiment
```

### 访问地址
- 人类实验: `/main.php`
- RL控制面板: `/rl_dashboard.html`
- RL API: `/rl_api.php`

## Neural Q-Learning System (神经网络增强Q-Learning)

### 概述
使用 LSTM + Self-Attention 的神经网络架构，从真实人类被试数据中学习决策模式。

### 文件结构
```
experiment/rl_agents/neural_ql/
├── data_loader.py      # 数据加载和预处理
├── model.py            # 神经网络模型定义
├── train.py            # 训练脚本
├── neural_ql_agent.py  # 实验代理
├── evaluate.py         # 模型评估工具
├── trained_models/     # 保存的模型
└── README.md           # 详细文档
```

### 训练模型
```bash
source .venv/bin/activate
cd experiment/rl_agents/neural_ql
python train.py --epochs 100 --batch_size 32
```

### 评估模型
```bash
python evaluate.py --visualize
```

### 人类数据放置
将人类被试CSV数据放在 `results/` 目录下，模型会自动加载。

## Recent Changes
- 2025-12-01: 添加神经网络增强Q-Learning系统
  - LSTM + Self-Attention 架构
  - 行为克隆训练从人类数据学习
  - 验证准确率达 86.30%
  - 集成到 RL Dashboard

- 2025-11-27: 添加RL系统
  - 创建 rl_api.php API端点
  - 创建 rl_dashboard.html 控制面板
  - 添加多个RL模型 (simple_ql, random_agent, ucb_agent)
  - 创建 results_rl/ 存储RL结果
  - 创建 custom_sequences/ 自定义序列示例
  - 添加详细文档 (rl_agents/README.md, custom_sequences/README.md)

- 2025-11-26: 初始化Replit环境
  - 安装PHP 8.2
  - 配置工作流
  - 安全改进

## 文件说明

| 目录/文件 | 用途 |
|-----------|------|
| experiment/rl_agents/README.md | RL模型开发指南 |
| experiment/custom_sequences/README.md | 自定义序列开发指南 |
| experiment/rl_agents/models/base_model.py | RL模型模板 |
| experiment/custom_sequences/static/example_biased.php | 静态序列示例 |
| experiment/custom_sequences/dynamic/example_wsls.py | 动态序列示例 |

## Notes
- RL结果与人类结果分开存储，便于对比分析
- 所有模型和序列支持通过URL参数切换，无需修改代码
- Q表等状态文件存储在 `rl_agents/q_tables/`
