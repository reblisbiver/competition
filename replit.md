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

## Recent Changes
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
