# Custom Sequences - 自定义序列分配方案

本文件夹用于存放自定义的奖励分配序列，分为静态(static)和动态(dynamic)两种类型。

## 目录结构

```
custom_sequences/
├── README.md           # 本说明文件
├── static/             # 静态序列 (PHP)
│   ├── example_biased.php
│   └── example_alternating.php
└── dynamic/            # 动态序列 (Python)
    └── example_wsls.py
```

## 如何使用

### 通过URL调用自定义序列

```
# 使用自定义静态序列
/rl_api.php?action=run_all&model=simple_ql&schedule_type=STATIC&schedule_name=example_biased

# 使用自定义动态序列
/rl_api.php?action=run_all&model=simple_ql&schedule_type=DYNAMIC&schedule_name=example_wsls
```

## 创建静态序列 (PHP)

静态序列预先定义好每个试次的奖励值，适合需要精确控制实验条件的场景。

### 文件格式

```php
<?php
// 必须定义这两个数组，每个包含100个元素(0或1)
$biased_rewards = [1, 0, 1, 0, ...];    // 偏向侧奖励
$unbiased_rewards = [0, 1, 0, 1, ...];  // 非偏向侧奖励
?>
```

### 示例：70%偏向概率

```php
<?php
$biased_rewards = [];
$unbiased_rewards = [];

for ($i = 0; $i < 100; $i++) {
    $biased_rewards[] = (rand(1, 100) <= 70) ? 1 : 0;
    $unbiased_rewards[] = (rand(1, 100) <= 30) ? 1 : 0;
}
?>
```

## 创建动态序列 (Python)

动态序列根据参与者的选择历史实时计算奖励，适合自适应实验设计。

### 输入参数

脚本通过命令行参数接收4个JSON编码的值：
1. `sys.argv[1]`: 偏向侧奖励历史 `[0, 1, 0, ...]`
2. `sys.argv[2]`: 非偏向侧奖励历史 `[1, 0, 1, ...]`
3. `sys.argv[3]`: 选择历史 `["True", "False", ...]` (True=选择偏向侧)
4. `sys.argv[4]`: 用户ID

### 输出格式

必须打印两个逗号分隔的值：
```
biased_reward, unbiased_reward
```

例如：`1, 0` 或 `0, 1`

### 模板

```python
#!/usr/bin/env python3
import sys
import json
import random

def main():
    # 解析输入
    bias_rewards = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    unbias_rewards = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
    choices = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
    user_id = sys.argv[4] if len(sys.argv) > 4 else "unknown"
    
    # 你的逻辑：根据历史计算本次奖励
    biased_reward = 1 if random.random() < 0.5 else 0
    unbiased_reward = 1 if random.random() < 0.5 else 0
    
    # 输出结果（格式必须正确！）
    print(f"{biased_reward}, {unbiased_reward}")

if __name__ == "__main__":
    main()
```

## 注意事项

1. **文件名即为schedule_name**: 创建 `my_schedule.php` 后，使用 `schedule_name=my_schedule` 调用
2. **静态序列必须定义100个元素**: 数组长度应与实验试次数匹配
3. **动态序列输出格式严格**: 只能输出一行，格式为 `数字, 数字`
4. **不要在Python脚本中打印调试信息**: 所有输出都会被解析为奖励值
5. **自定义序列优先**: 同名时，custom_sequences/ 下的文件优先于 sequences/ 下的默认文件
