<?php
session_start();


// Helper: normalize array elements to integers (0/1) or booleans where appropriate
function normalize_rewards_array($arr) {
    if (!is_array($arr)) return array();
    $out = array();
    foreach ($arr as $el) {
        if ($el === null) { $out[] = 0; continue; }
        if ($el === '') { $out[] = 0; continue; }
        if (is_bool($el)) { $out[] = $el ? 1 : 0; continue; }
        if (is_numeric($el)) { $out[] = intval($el); continue; }
        $low = strtolower(strval($el));
        if ($low === 'true' || $low === '1') { $out[] = 1; continue; }
        if ($low === 'false' || $low === '0') { $out[] = 0; continue; }
        // Fallback: empty -> 0
        $out[] = 0;
    }
    return $out;
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Last update: 21.7.2019
//
// Thanks to Amir Dezfouli and Tsahi Asher for their help in improving this code.
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++ 新增：模式判断 +++++++++++++++
$mode = $_GET['mode'] ?? 'human'; // 默认人类模式
$user_id = $_SESSION['user_id'];  // 原有user_id，用于RL的Q表区分
// +++++++++++++++++++++++++++++++++++++++++++++

// 新增：初始化LEFT/RIGHT选择次数的SESSION变量（避免未定义）
if (!isset($_SESSION['left_count'])) {
    $_SESSION['left_count'] = 0;
}
if (!isset($_SESSION['right_count'])) {
    $_SESSION['right_count'] = 0;
}

$biased_side = 'biased_side';
$unbiased_side = 'unbiased_side';

/////////////////////////////////////////////////////////////////////////////////

if (isset($_GET['NUMBER_OF_TRIALS'])) {
    // 转换为整数类型，确保后续判断正确
    $_SESSION['NUMBER_OF_TRIALS'] = intval($_GET['NUMBER_OF_TRIALS']);
}

// Get choice in current trial
/////////////////////////////////////////////////////////////////////////////////
$is_biased_side = $_SESSION[$biased_side]==$_GET["side"];
$is_biased_choice = $is_biased_side ? 'True' : 'False';

// +++++++++++++++ 新增：根据模式获取动作 +++++++++++++++
$side = '';
if ($mode == 'ql') {
    // RL模式：调用Python脚本获取动作
    $last_action = $_SESSION['rl_last_action'] ?? 'None'; // 上一次动作（初始为None）
    $last_reward = $_SESSION['rl_last_reward'] ?? '0';    // 上一次奖励（初始为0）

    // 拼接Python调用命令（注意路径！__DIR__是scripts/，所以要回退到rl_agents）
    $python_script = __DIR__ . '/../rl_agents/simple_ql.py';
    $command = "python3 {$python_script} {$user_id} {$last_action} {$last_reward}";
    $side = trim(exec($command)); // 拿到RL选的动作（LEFT/RIGHT）

    // 保存本次动作到SESSION，下次RL调用用
    $_SESSION['rl_last_action'] = $side;
} else {
    // 人类模式：用前端传的side（原有逻辑）
    $side = $_GET["side"];
}
// +++++++++++++++++++++++++++++++++++++++++++++

// NOTE: 不再在此处即时累加 left/right 次数（可能导致重复计数或与试次数不同步）
// 计数将在试次结束时根据 $_SESSION['is_bias_choice'] 重新计算，保证一致性。

/////////////////////////////////////////////////////////////////////////////////
// Allocate rewards to next trial
/////////////////////////////////////////////////////////////////////////////////
$result_string = "";
    /////////////////////////////////////////////////////////////////////////////
    // DYNAMIC Allocation of rewards to next trial
    /////////////////////////////////////////////////////////////////////////////
$current_biased_reward = $current_unbiased_reward = NULL;
// The name of the dynamic schedule should be set in the session parameter, e.g.:
//       $_SESSION['schedule_name'] = "my_dynamic_model";
// The name should refer to a python file that exists in sequences/dynamic folder
// A good place for this definition is at main.php
if ($_SESSION['schedule_type'] == "DYNAMIC") { // Game is dynamic
    // Build full script path and check existence
    $script_path = __DIR__ . '/../sequences/dynamic/' . $_SESSION['schedule_name'] . '.py';
    // Prepare command safely using escapeshellarg
    $python_exec = 'python'; // adjust to 'python3' if your system requires it
    $arg_bias = json_encode($_SESSION['bias_rewards']);
    $arg_unbias = json_encode($_SESSION['unbias_rewards']);
    $arg_isbias = json_encode($_SESSION['is_bias_choice']);
    $arg_user = json_encode($_SESSION['user_id']);

    // Prefer calling a persistent inference service to avoid Python startup overhead
    // Production: use the provided Replit URL. Locally you can set this to http://127.0.0.1:8001/predict
    $service_url = 'https://5231b921-c6e0-4070-a57a-99f8c47c409a-00-1fg3di8hyqdb7.pike.replit.dev/predict';
    // include `model` so the service can select which model implementation to use
    $payload = array(
        'bias_rewards' => $_SESSION['bias_rewards'] ?? array(),
        'unbias_rewards' => $_SESSION['unbias_rewards'] ?? array(),
        'is_bias_choice' => $_SESSION['is_bias_choice'] ?? array(),
        'user_id' => $_SESSION['user_id'] ?? '',
        'model' => $_SESSION['schedule_name'] ?? ''
    );
    $json_payload = json_encode($payload);
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $service_url);
    curl_setopt($ch, CURLOPT_POST, 1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $json_payload);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
    curl_setopt($ch, CURLOPT_TIMEOUT, 5); // short timeout; fall back if service not available

    $svc_resp = curl_exec($ch);
    $curl_errno = curl_errno($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($svc_resp !== false && $curl_errno == 0 && $http_code >= 200 && $http_code < 300) {
        $j = json_decode($svc_resp, true);
        // Expect new keys ('biased'/'unbiased') or legacy keys ('target_alloc'/'anti_alloc')
        if (is_array($j) && isset($j['biased']) && isset($j['unbiased'])) {
            $current_biased_reward = intval($j['biased']);
            $current_unbiased_reward = intval($j['unbiased']);
        } elseif (is_array($j) && isset($j['target_alloc']) && isset($j['anti_alloc'])) {
            $current_biased_reward = intval($j['target_alloc']);
            $current_unbiased_reward = intval($j['anti_alloc']);
        } else {
            error_log('ERROR: inference service returned unexpected payload: ' . $svc_resp);
            header('Content-Type: application/json', true, 502);
            echo json_encode(array('status' => 'error', 'message' => 'Inference service returned unexpected payload', 'payload' => $svc_resp));
            exit;
        }
    } else {
        error_log('ERROR: inference service unreachable or error (errno:' . $curl_errno . ' code:' . $http_code . ')');
        header('Content-Type: application/json', true, 502);
        echo json_encode(array('status' => 'error', 'message' => 'Inference service unreachable', 'errno' => $curl_errno, 'http_code' => $http_code));
        exit;
    }
}
    /////////////////////////////////////////////////////////////////////////////
    // STATIC - Allocation of rewards to next trial
    /////////////////////////////////////////////////////////////////////////////
else{ // Game is static
    include '../sequences/static/' . $_SESSION['schedule_name'] . '.php';
    $current_biased_reward = $biased_rewards[$_SESSION['trial_number']];
    $current_unbiased_reward = $unbiased_rewards[$_SESSION['trial_number']];
}
if ($is_biased_side){ // If current choice was of the biased side
    $current_reward = $current_biased_reward;
    $current_unobserved_reward = $current_unbiased_reward;
}
else{
    $current_reward = $current_unbiased_reward;
    $current_unobserved_reward = $current_biased_reward;
}
array_push($_SESSION['bias_rewards'],$current_biased_reward);
array_push($_SESSION['unbias_rewards'],$current_unbiased_reward);
array_push($_SESSION['is_bias_choice'],$is_biased_choice);

/////////////////////////////////////////////////////////////////////////////////
// Write current trial's data
/////////////////////////////////////////////////////////////////////////////////
function remove_last_char_from_file($file_name){
    $fh = fopen($file_name, 'r+') or die("can't open file");
    $stat = fstat($fh);
    ftruncate($fh, $stat['size']-1);
    fclose($fh);
}

function write_trial_data_csv($is_biased_choice, $current_reward, $current_unobserved_reward, $path){
    $file_name = $path . $_SESSION['user_id'] . '.csv';
    if (!file_exists($file_name)){
        file_put_contents($file_name,
            "trial_number, time, schedule_type, schedule_name, is_biased_choice, side_choice, RT, observed_reward, unobserved_reward, biased_reward, unbiased_reward" .PHP_EOL , FILE_APPEND);
    }
    global $current_biased_reward, $current_unbiased_reward;
    date_default_timezone_set('Asia/Shanghai');
    $trial_data =
        $_SESSION['trial_number'] . ','
        . date("Y-m-d h:i:sa") . ', '
        . $_SESSION['schedule_type'] . ', '
        . $_SESSION['schedule_name'] . ', '
        . strtolower($is_biased_choice) . ', '
        . $_GET['side']  . ', '
        . $_GET["RT"] . ', '
        . $current_reward . ', '
        . $current_unobserved_reward . ', '
        . $current_biased_reward . ', '
        . $current_unbiased_reward
        . PHP_EOL;
    file_put_contents($file_name, $trial_data, FILE_APPEND);
}
 $path = __DIR__ . "/../../real_results/"; // unified results folder at workspace root (merged storage)
 if (!file_exists($path)) { // Create the real_results directory if it doesn't exist
     mkdir($path, 0777, true);
 }
write_trial_data_csv($is_biased_choice, $current_reward, $current_unobserved_reward, $path);

// +++++++++++++++ 新增代码：判断是否为最后一次试次，追加总和 +++++++++++++++
// 假设总试次数为100次，试次编号从0到99（共100次）
$total_trials = $_SESSION['NUMBER_OF_TRIALS'];
if ($_SESSION['trial_number'] == $total_trials - 1) { 
    $file_name = $path . $_SESSION['user_id'] . '.csv';
    // 构造总和行：total, user_id, schedule_type, schedule_name,
    //               biased_side, biased_count, unbiased_side, unbiased_count, rewards, total_reward
    $user_id = isset($_SESSION['user_id']) ? $_SESSION['user_id'] : 'Error_NoUserID';
    $sched_type = isset($_SESSION['schedule_type']) ? $_SESSION['schedule_type'] : 'Error_NoSchedType';
    $sched_name = isset($_SESSION['schedule_name']) ? $_SESSION['schedule_name'] : 'Error_NoSchedName';
    // Determine biased/unbiased counts based on which side was assigned biased in session
    $biased_side_name = isset($_SESSION[$biased_side]) ? $_SESSION[$biased_side] : 'LEFT';
    $unbiased_side_name = isset($_SESSION[$unbiased_side]) ? $_SESSION[$unbiased_side] : 'RIGHT';
    // Recompute left/right counts from is_bias_choice array to ensure consistency
    $is_bias_arr = isset($_SESSION['is_bias_choice']) && is_array($_SESSION['is_bias_choice']) ? $_SESSION['is_bias_choice'] : array();
    $count_bias_true = 0;
    $count_bias_false = 0;
    foreach ($is_bias_arr as $v) {
        $val = strtolower(strval($v));
        if ($val === 'true' || $v === 1 || $v === '1') {
            $count_bias_true++;
        } elseif ($val === 'false' || $v === 0 || $v === '0') {
            $count_bias_false++;
        }
        // ignore other/invalid entries
    }
    // Map to biased/unbiased and left/right depending on which side was marked biased in session
    if ($biased_side_name === 'LEFT') {
        $biased_count = $count_bias_true;
        $unbiased_count = $count_bias_false;
        $left_count = $biased_count;
        $right_count = $unbiased_count;
    } else {
        $biased_count = $count_bias_true;
        $unbiased_count = $count_bias_false;
        $right_count = $biased_count;
        $left_count = $unbiased_count;
    }
    
    $total_reward = isset($_SESSION['total_reward']) ? intval($_SESSION['total_reward']) : 0;

    $total_line = "total, "
        . $user_id . ", "
        . $sched_type . ", "
        . $sched_name . ", "
        . "biased_side" . ", "
        . $biased_count . ", "
        . "unbiased_side" . ", "
        . $unbiased_count . ", "
        . "rewards, " . $total_reward
        . PHP_EOL;
    file_put_contents($file_name, $total_line, FILE_APPEND);
    // 更新 schedules_counts.json 中对应 schedule 的调用计数（最小实现）
    $counts_file = __DIR__ . '/../schedules_counts.json';
    $counts = array();
    if (file_exists($counts_file)){
        $raw = file_get_contents($counts_file);
        $decoded = json_decode($raw, true);
        if (is_array($decoded)) $counts = $decoded;
    }
    $sname = isset($_SESSION['schedule_name']) ? $_SESSION['schedule_name'] : '';
    if ($sname !== ''){
        if (!isset($counts[$sname])) $counts[$sname] = 0;
        $counts[$sname] = intval($counts[$sname]) + 1;
        file_put_contents($counts_file, json_encode($counts, JSON_PRETTY_PRINT));
        // 另外写一行简单日志，便于人工查看
        $log_file = __DIR__ . '/../schedules_calls.log';
        $log_line = date('c') . "\t" . $sname . "\n";
        file_put_contents($log_file, $log_line, FILE_APPEND);
        error_log('DEBUG: incremented schedule count for ' . $sname);
    } else {
        error_log('WARNING: schedule_name empty when attempting to increment count');
    }
}
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/////////////////////////////////////////////////////////////////////////////////
// Manage the game
/////////////////////////////////////////////////////////////////////////////////

// +++++++++++++++ 新增：RL模式下保存本次奖励 +++++++++++++++
if ($mode == 'ql') {
    $_SESSION['rl_last_reward'] = $current_reward; // 存本次奖励，下次RL更新用
}
// +++++++++++++++++++++++++++++++++++++++++++++

if ($current_reward){
    $_SESSION["total_reward"]++;
}

$_SESSION['trial_number'] = $_SESSION['trial_number'] + 1;
echo $current_reward;
?>
