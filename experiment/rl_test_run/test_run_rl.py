import requests
import time
import os

# -------------------------- 配置项 --------------------------
BASE_URL = "https://5231b921-c6e0-4070-a57a-99f8c47c409a-00-1fg3di8hyqdb7.pike.replit.dev/scripts/backend.php"  # 替换成你的后端URL
NUMBER_OF_TRIALS = 10  # 总试次数（和前端NUMBER_OF_TRIALS一致）
RL_USER_ID = "test_rl_001"  # 测试用的user_id（可选，PHP会自动生成，这里指定方便跟踪）
# ------------------------------------------------------------

# 创建Session维持会话（关键！PHP的Session靠Cookie维持）
session = requests.Session()

# 初始化：先请求一次main.php获取Session（可选，确保user_id生成）
session.get(
    "https://5231b921-c6e0-4070-a57a-99f8c47c409a-00-1fg3di8hyqdb7.pike.replit.dev/main.php"
)

# 可选：清理旧的Q表文件（避免上次测试残留）
q_table_path = f"./experiment/rl_agents/q_tables/q_{RL_USER_ID}.txt"
if os.path.exists(q_table_path):
    os.remove(q_table_path)
    print(f"已清理旧Q表文件：{q_table_path}")

# 循环执行RL试次
print(f"\n开始RL自动测试（共{NUMBER_OF_TRIALS}次试次）：")
print("-" * 50)

for trial_idx in range(1, NUMBER_OF_TRIALS + 1):
    # 构造请求参数
    params = {
        "mode": "ql",  # 启用RL模式
        "RT": 1000,  # 随便填，不影响RL逻辑
        "NUMBER_OF_TRIALS": NUMBER_OF_TRIALS,  # 传递总试次数给PHP
        "user_id": RL_USER_ID  # 可选，指定user_id方便跟踪
    }

    try:
        # 发送GET请求到后端
        response = session.get(BASE_URL, params=params, timeout=5)

        # 解析响应（后端返回的是本次奖励值：0或1）
        current_reward = response.text.strip()
        # 从Session中获取RL选的动作（PHP存在SESSION['rl_last_action']）
        # 注意：PHP的Session数据前端拿不到，这里可以打印响应Cookie或日志辅助跟踪
        print(
            f"试次 {trial_idx:2d} | RL选择动作：{session.cookies.get('rl_last_action')} | 奖励：{current_reward}"
        )

        # 模拟试次间隔（可选）
        time.sleep(0.5)

    except requests.exceptions.RequestException as e:
        print(f"试次 {trial_idx} 请求失败：{e}")
        break

print("-" * 50)
print("RL自动测试结束！")
print(f"\n查看结果：")
print(f"- Q表文件：./experiment/rl_agents/q_tables/q_{RL_USER_ID}.txt")
print(
    f"- CSV数据：./experiment/results/[schedule_type]/[schedule_name]/{RL_USER_ID}.csv"
)
