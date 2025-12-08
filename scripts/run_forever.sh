#!/usr/bin/env bash  
# Usage: run_forever.sh <RUN-COMMAND> ...

# 保存业务进程 PID，用于监控
RUN_PID=""

# 启动/重启业务进程的函数
start_business() {
    echo "=== 启动业务进程: $* ==="
    "$@" &  # 修正：用 "$@" 替代 $*，正确传递带空格的参数
    RUN_PID=$!  # 记录业务进程PID
    echo "=== 业务进程 PID: $RUN_PID ==="
}

# 第一步：启动业务进程
start_business "$@"

# start server in background for HTTP requests to keep alive
python3 -c "from http.server import BaseHTTPRequestHandler, HTTPServer

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello there')

    def do_OPTIONS(self):
      self.send_response(200, 'ok')
      self.send_header('Access-Control-Allow-Origin', '*')
      self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
      self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
      self.end_headers()

server_address = ('0.0.0.0', 8000)
httpd = HTTPServer(server_address, MyServer)
httpd.serve_forever()" &
SERVER_PID=$!  # 记录保活服务器PID
echo "=== 保活服务器启动，PID: $SERVER_PID ==="

# 主循环：监控业务进程 + 定时请求你的固定网址
while true
do
  # 新增：检查业务进程是否崩溃，崩溃则重启
  if ! kill -0 $RUN_PID 2>/dev/null; then
      echo "=== 业务进程已崩溃，正在重启... ==="
      start_business "$@"
  fi

  # 保留你的固定网址，不修改
  curl -s https://5231b921-c6e0-4070-a57a-99f8c47c409a-00-1fg3di8hyqdb7.pike.replit > /dev/null
  
  # 可选：输出日志，方便查看保活状态
  echo "=== 保活请求发送成功，等待3分钟后继续 ==="
  sleep 180
done