<?php
session_start();
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

// 新增：根据当前选择的side，更新LEFT/RIGHT计数
if ($_GET['side'] === 'LEFT') {
    $_SESSION['left_count']++;
} elseif ($_GET['side'] === 'RIGHT') {
    $_SESSION['right_count']++;
}

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
if($_SESSION['schedule_type'] == "DYNAMIC"){ //Game is dynamic
    $run_python_command = 'python ../sequences/dynamic/'. $_SESSION['schedule_name'] . '.py '
        . json_encode($_SESSION['bias_rewards']) . ' '
        . json_encode($_SESSION['unbias_rewards']) . ' '
        . json_encode($_SESSION['is_bias_choice']) . ' '
        . json_encode($_SESSION['user_id']);
    $result_string = exec($run_python_command);
    $result_array = explode(", ", $result_string);
    $current_biased_reward = $result_array[0];
    $current_unbiased_reward = $result_array[1];
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
$path = __DIR__ . "/../../results/" . $_SESSION['schedule_type'] . "/" . $_SESSION['schedule_name'] . "/";
if (!file_exists($path)) { // Create the results directory for current mechanism if it doesn't exist yest
    mkdir($path, 0777, true);
}
write_trial_data_csv($is_biased_choice, $current_reward, $current_unobserved_reward, $path);

// +++++++++++++++ 新增代码：判断是否为最后一次试次，追加总和 +++++++++++++++
// 假设总试次数为100次，试次编号从0到99（共100次）
$total_trials = $_SESSION['NUMBER_OF_TRIALS'];
if ($_SESSION['trial_number'] == $total_trials - 1) { 
    $file_name = $path . $_SESSION['user_id'] . '.csv';
    // 新增：拼接LEFT/RIGHT选择次数（格式：left_count/right_count）
    $side_choice_stats = $_SESSION['left_count'] . '/' . $_SESSION['right_count'];
    // 构造总和行（与CSV现有列对应，仅填充必要字段）
    $total_line = "total, , " . 
      $_SESSION['schedule_type'] . ", " . 
      $_SESSION['schedule_name'] . ", , " . 
      $side_choice_stats . ", , " .  // side_choice列填LEFT/RIGHT次数
      $_SESSION["total_reward"] . ", , , " . 
      PHP_EOL;
    file_put_contents($file_name, $total_line, FILE_APPEND);
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
