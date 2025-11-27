# RL Agents - 强化学习代理

本文件夹包含用于选择实验的强化学习代理。

## 目录结构

```
rl_agents/
├── README.md           # 本说明文件
├── simple_ql.py        # 默认Q-Learning代理
├── q_tables/           # Q表存储目录
│   └── q_*.txt         # 每个session的Q表
└── models/             # 自定义模型目录
    ├── base_model.py   # 模型模板
    ├── random_agent.py # 随机代理（基线）
    └── ucb_agent.py    # UCB代理
```

## 快速开始

### 通过URL运行RL实验

```bash
# 运行完整实验（100个试次）
curl "https://your-domain/rl_api.php?action=run_all&model=simple_ql&schedule_type=STATIC&schedule_name=random_0"

# 初始化会话
curl "https://your-domain/rl_api.php?action=init&model=simple_ql&trials=50"

# 单步执行
curl "https://your-domain/rl_api.php?action=run_trial"

# 查看可用模型
curl "https://your-domain/rl_api.php?action=list_models"

# 查看可用序列
curl "https://your-domain/rl_api.php?action=list_schedules"
```

## API参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| action | 操作类型: init/run_trial/run_all/status/list_models/list_schedules | run_trial |
| model | RL模型名称 | simple_ql |
| schedule_type | 序列类型: STATIC/DYNAMIC | STATIC |
| schedule_name | 序列名称 | random_0 |
| trials | 试次数量 | 100 |

## 如何添加新的RL模型

### 1. 创建Python脚本

在 `models/` 目录下创建新文件，例如 `my_model.py`：

```python
#!/usr/bin/env python3
"""
My Custom RL Model
"""
import sys
import os

def main():
    # 解析命令行参数
    user_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    last_action = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ["None", "null"] else None
    try:
        last_reward = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ["None", "null"] else 0.0
    except:
        last_reward = 0.0
    
    # 你的决策逻辑
    action = "LEFT"  # 或 "RIGHT"
    
    # 必须打印动作（只能是 LEFT 或 RIGHT）
    print(action)

if __name__ == "__main__":
    main()
```

### 2. 接口规范

**输入参数** (通过命令行):
- `sys.argv[1]`: user_id - 用户标识符
- `sys.argv[2]`: last_action - 上一次动作 ("LEFT"/"RIGHT"/"None")
- `sys.argv[3]`: last_reward - 上一次奖励 ("0"/"1"/"None")

**输出要求**:
- 必须且只能打印 `LEFT` 或 `RIGHT`
- 不要打印任何其他内容（调试信息请用stderr）

### 3. 状态持久化

如需保存模型状态，使用 `q_tables/` 目录：

```python
import os

STATE_DIR = os.path.join(os.path.dirname(__file__), "../q_tables")
os.makedirs(STATE_DIR, exist_ok=True)

state_file = os.path.join(STATE_DIR, f"mymodel_{user_id}.txt")
```

### 4. 使用新模型

```
/rl_api.php?action=run_all&model=my_model
```

## 内置模型

### simple_ql (默认)
- 简单Q-Learning
- ε-greedy策略 (ε=0.1)
- 学习率 α=0.1

### random_agent
- 随机选择
- 50/50概率

### ucb_agent
- Upper Confidence Bound
- 自动平衡探索与利用

## 结果存储

RL实验结果保存在 `results_rl/` 目录：
```
results_rl/
└── STATIC/
    └── random_0/
        └── simple_ql/
            └── rl_1234567890_42.csv
```

CSV格式：
```
trial_number,time,schedule_type,schedule_name,model,is_biased_choice,side_choice,observed_reward,unobserved_reward,biased_reward,unbiased_reward,total_reward
```

## 调试技巧

1. **测试模型脚本**:
```bash
python3 models/my_model.py test_user None None
```

2. **检查输出**:
```bash
python3 models/my_model.py test_user LEFT 1
# 应该输出: LEFT 或 RIGHT
```

3. **查看错误日志**:
```bash
python3 models/my_model.py test_user LEFT 1 2>&1
```
