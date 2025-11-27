<?php
session_start();
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');

$response = ['success' => false, 'error' => '', 'data' => []];

$action = $_GET['action'] ?? 'run_trial';
$rl_model = $_GET['model'] ?? 'simple_ql';
$schedule_type = strtoupper($_GET['schedule_type'] ?? 'STATIC');
$schedule_name = $_GET['schedule_name'] ?? 'random_0';
$number_of_trials = intval($_GET['trials'] ?? 100);

if ($action === 'init') {
    $_SESSION['rl_user_id'] = 'rl_' . time() . '_' . rand(1, 100);
    $_SESSION['rl_model'] = $rl_model;
    $_SESSION['rl_schedule_type'] = $schedule_type;
    $_SESSION['rl_schedule_name'] = $schedule_name;
    $_SESSION['rl_number_of_trials'] = $number_of_trials;
    $_SESSION['rl_trial_number'] = 0;
    $_SESSION['rl_total_reward'] = 0;
    $_SESSION['rl_last_action'] = null;
    $_SESSION['rl_last_reward'] = null;
    $_SESSION['rl_bias_rewards'] = [];
    $_SESSION['rl_unbias_rewards'] = [];
    $_SESSION['rl_is_bias_choice'] = [];
    $_SESSION['rl_left_count'] = 0;
    $_SESSION['rl_right_count'] = 0;
    
    if (rand(1,2) == 1) {
        $_SESSION['rl_biased_side'] = 'LEFT';
        $_SESSION['rl_unbiased_side'] = 'RIGHT';
    } else {
        $_SESSION['rl_biased_side'] = 'RIGHT';
        $_SESSION['rl_unbiased_side'] = 'LEFT';
    }
    
    $response['success'] = true;
    $response['data'] = [
        'user_id' => $_SESSION['rl_user_id'],
        'model' => $rl_model,
        'schedule_type' => $schedule_type,
        'schedule_name' => $schedule_name,
        'total_trials' => $number_of_trials,
        'biased_side' => $_SESSION['rl_biased_side'],
        'message' => 'Session initialized. Call ?action=run_trial to start.'
    ];
    echo json_encode($response);
    exit;
}

if ($action === 'run_trial') {
    if (!isset($_SESSION['rl_user_id'])) {
        $response['error'] = 'Session not initialized. Call ?action=init first.';
        echo json_encode($response);
        exit;
    }
    
    if ($_SESSION['rl_trial_number'] >= $_SESSION['rl_number_of_trials']) {
        $response['success'] = true;
        $response['data'] = [
            'status' => 'completed',
            'total_reward' => $_SESSION['rl_total_reward'],
            'total_trials' => $_SESSION['rl_number_of_trials'],
            'left_count' => $_SESSION['rl_left_count'],
            'right_count' => $_SESSION['rl_right_count'],
            'message' => 'All trials completed!'
        ];
        echo json_encode($response);
        exit;
    }
    
    $user_id = $_SESSION['rl_user_id'];
    $model = $_SESSION['rl_model'];
    $last_action = $_SESSION['rl_last_action'] ?? 'None';
    $last_reward = $_SESSION['rl_last_reward'] ?? '0';
    
    $model_script = __DIR__ . "/rl_agents/models/{$model}.py";
    if (!file_exists($model_script)) {
        $model_script = __DIR__ . "/rl_agents/simple_ql.py";
    }
    
    $command = "python3 {$model_script} " . escapeshellarg($user_id) . " " . escapeshellarg($last_action) . " " . escapeshellarg($last_reward) . " 2>&1";
    $rl_action = trim(exec($command, $output, $return_code));
    
    if ($return_code !== 0 || !in_array($rl_action, ['LEFT', 'RIGHT'])) {
        $response['error'] = "RL model error: " . implode("\n", $output);
        echo json_encode($response);
        exit;
    }
    
    $_SESSION['rl_last_action'] = $rl_action;
    
    if ($rl_action === 'LEFT') {
        $_SESSION['rl_left_count']++;
    } else {
        $_SESSION['rl_right_count']++;
    }
    
    $is_biased_side = ($_SESSION['rl_biased_side'] === $rl_action);
    $is_biased_choice = $is_biased_side ? 'True' : 'False';
    
    $current_biased_reward = $current_unbiased_reward = 0;
    $schedule_type = $_SESSION['rl_schedule_type'];
    $schedule_name = $_SESSION['rl_schedule_name'];
    
    $custom_static_path = __DIR__ . "/custom_sequences/static/{$schedule_name}.php";
    $custom_dynamic_path = __DIR__ . "/custom_sequences/dynamic/{$schedule_name}.py";
    $default_static_path = __DIR__ . "/sequences/static/{$schedule_name}.php";
    
    if ($schedule_type === 'DYNAMIC') {
        if (file_exists($custom_dynamic_path)) {
            $script_path = $custom_dynamic_path;
        } else {
            $script_path = __DIR__ . "/sequences/dynamic/{$schedule_name}.py";
        }
        
        $run_python_command = "python3 {$script_path} "
            . escapeshellarg(json_encode($_SESSION['rl_bias_rewards'])) . ' '
            . escapeshellarg(json_encode($_SESSION['rl_unbias_rewards'])) . ' '
            . escapeshellarg(json_encode($_SESSION['rl_is_bias_choice'])) . ' '
            . escapeshellarg($user_id) . " 2>&1";
        $result_string = exec($run_python_command);
        $result_array = explode(", ", $result_string);
        $current_biased_reward = intval($result_array[0] ?? 0);
        $current_unbiased_reward = intval($result_array[1] ?? 0);
    } else {
        if (file_exists($custom_static_path)) {
            include $custom_static_path;
        } elseif (file_exists($default_static_path)) {
            include $default_static_path;
        } else {
            $biased_rewards = array_fill(0, 100, 0);
            $unbiased_rewards = array_fill(0, 100, 0);
        }
        $trial_idx = $_SESSION['rl_trial_number'];
        $current_biased_reward = $biased_rewards[$trial_idx] ?? 0;
        $current_unbiased_reward = $unbiased_rewards[$trial_idx] ?? 0;
    }
    
    if ($is_biased_side) {
        $current_reward = $current_biased_reward;
        $current_unobserved_reward = $current_unbiased_reward;
    } else {
        $current_reward = $current_unbiased_reward;
        $current_unobserved_reward = $current_biased_reward;
    }
    
    array_push($_SESSION['rl_bias_rewards'], $current_biased_reward);
    array_push($_SESSION['rl_unbias_rewards'], $current_unbiased_reward);
    array_push($_SESSION['rl_is_bias_choice'], $is_biased_choice);
    
    $_SESSION['rl_last_reward'] = $current_reward;
    
    $path = __DIR__ . "/../results_rl/{$schedule_type}/{$schedule_name}/{$model}/";
    if (!file_exists($path)) {
        mkdir($path, 0777, true);
    }
    
    $file_name = $path . $user_id . '.csv';
    if (!file_exists($file_name)) {
        $header = "trial_number,time,schedule_type,schedule_name,model,is_biased_choice,side_choice,observed_reward,unobserved_reward,biased_reward,unbiased_reward,total_reward\n";
        file_put_contents($file_name, $header);
    }
    
    date_default_timezone_set('Asia/Shanghai');
    $trial_data = $_SESSION['rl_trial_number'] . ','
        . date("Y-m-d H:i:s") . ','
        . $schedule_type . ','
        . $schedule_name . ','
        . $model . ','
        . strtolower($is_biased_choice) . ','
        . $rl_action . ','
        . $current_reward . ','
        . $current_unobserved_reward . ','
        . $current_biased_reward . ','
        . $current_unbiased_reward . ','
        . $_SESSION['rl_total_reward'] . "\n";
    file_put_contents($file_name, $trial_data, FILE_APPEND);
    
    if ($current_reward) {
        $_SESSION['rl_total_reward']++;
    }
    
    $_SESSION['rl_trial_number']++;
    
    $is_last_trial = ($_SESSION['rl_trial_number'] >= $_SESSION['rl_number_of_trials']);
    
    if ($is_last_trial) {
        $total_line = "total,,"
            . $schedule_type . ","
            . $schedule_name . ","
            . $model . ",,"
            . $_SESSION['rl_left_count'] . "/" . $_SESSION['rl_right_count'] . ","
            . $_SESSION['rl_total_reward'] . ",,,,\n";
        file_put_contents($file_name, $total_line, FILE_APPEND);
    }
    
    $response['success'] = true;
    $response['data'] = [
        'trial' => $_SESSION['rl_trial_number'],
        'total_trials' => $_SESSION['rl_number_of_trials'],
        'action' => $rl_action,
        'reward' => $current_reward,
        'total_reward' => $_SESSION['rl_total_reward'],
        'is_biased_choice' => $is_biased_choice,
        'status' => $is_last_trial ? 'completed' : 'in_progress',
        'left_count' => $_SESSION['rl_left_count'],
        'right_count' => $_SESSION['rl_right_count']
    ];
    echo json_encode($response);
    exit;
}

if ($action === 'run_all') {
    $_SESSION['rl_user_id'] = 'rl_' . time() . '_' . rand(1, 100);
    $_SESSION['rl_model'] = $rl_model;
    $_SESSION['rl_schedule_type'] = $schedule_type;
    $_SESSION['rl_schedule_name'] = $schedule_name;
    $_SESSION['rl_number_of_trials'] = $number_of_trials;
    $_SESSION['rl_trial_number'] = 0;
    $_SESSION['rl_total_reward'] = 0;
    $_SESSION['rl_last_action'] = null;
    $_SESSION['rl_last_reward'] = null;
    $_SESSION['rl_bias_rewards'] = [];
    $_SESSION['rl_unbias_rewards'] = [];
    $_SESSION['rl_is_bias_choice'] = [];
    $_SESSION['rl_left_count'] = 0;
    $_SESSION['rl_right_count'] = 0;
    
    if (rand(1,2) == 1) {
        $_SESSION['rl_biased_side'] = 'LEFT';
        $_SESSION['rl_unbiased_side'] = 'RIGHT';
    } else {
        $_SESSION['rl_biased_side'] = 'RIGHT';
        $_SESSION['rl_unbiased_side'] = 'LEFT';
    }
    
    $user_id = $_SESSION['rl_user_id'];
    $model = $_SESSION['rl_model'];
    $all_trials = [];
    
    for ($t = 0; $t < $number_of_trials; $t++) {
        $last_action = $_SESSION['rl_last_action'] ?? 'None';
        $last_reward = $_SESSION['rl_last_reward'] ?? '0';
        
        $model_script = __DIR__ . "/rl_agents/models/{$model}.py";
        if (!file_exists($model_script)) {
            $model_script = __DIR__ . "/rl_agents/simple_ql.py";
        }
        
        $command = "python3 {$model_script} " . escapeshellarg($user_id) . " " . escapeshellarg($last_action) . " " . escapeshellarg($last_reward) . " 2>&1";
        $rl_action = trim(exec($command, $output, $return_code));
        
        if ($return_code !== 0 || !in_array($rl_action, ['LEFT', 'RIGHT'])) {
            $response['error'] = "RL model error at trial $t: " . implode("\n", $output);
            echo json_encode($response);
            exit;
        }
        
        $_SESSION['rl_last_action'] = $rl_action;
        
        if ($rl_action === 'LEFT') {
            $_SESSION['rl_left_count']++;
        } else {
            $_SESSION['rl_right_count']++;
        }
        
        $is_biased_side = ($_SESSION['rl_biased_side'] === $rl_action);
        $is_biased_choice = $is_biased_side ? 'True' : 'False';
        
        $current_biased_reward = $current_unbiased_reward = 0;
        
        $custom_static_path = __DIR__ . "/custom_sequences/static/{$schedule_name}.php";
        $custom_dynamic_path = __DIR__ . "/custom_sequences/dynamic/{$schedule_name}.py";
        $default_static_path = __DIR__ . "/sequences/static/{$schedule_name}.php";
        
        if ($schedule_type === 'DYNAMIC') {
            if (file_exists($custom_dynamic_path)) {
                $script_path = $custom_dynamic_path;
            } else {
                $script_path = __DIR__ . "/sequences/dynamic/{$schedule_name}.py";
            }
            
            $run_python_command = "python3 {$script_path} "
                . escapeshellarg(json_encode($_SESSION['rl_bias_rewards'])) . ' '
                . escapeshellarg(json_encode($_SESSION['rl_unbias_rewards'])) . ' '
                . escapeshellarg(json_encode($_SESSION['rl_is_bias_choice'])) . ' '
                . escapeshellarg($user_id) . " 2>&1";
            $result_string = exec($run_python_command);
            $result_array = explode(", ", $result_string);
            $current_biased_reward = intval($result_array[0] ?? 0);
            $current_unbiased_reward = intval($result_array[1] ?? 0);
        } else {
            if (file_exists($custom_static_path)) {
                include $custom_static_path;
            } elseif (file_exists($default_static_path)) {
                include $default_static_path;
            } else {
                $biased_rewards = array_fill(0, 100, 0);
                $unbiased_rewards = array_fill(0, 100, 0);
            }
            $current_biased_reward = $biased_rewards[$t] ?? 0;
            $current_unbiased_reward = $unbiased_rewards[$t] ?? 0;
        }
        
        if ($is_biased_side) {
            $current_reward = $current_biased_reward;
            $current_unobserved_reward = $current_unbiased_reward;
        } else {
            $current_reward = $current_unbiased_reward;
            $current_unobserved_reward = $current_biased_reward;
        }
        
        array_push($_SESSION['rl_bias_rewards'], $current_biased_reward);
        array_push($_SESSION['rl_unbias_rewards'], $current_unbiased_reward);
        array_push($_SESSION['rl_is_bias_choice'], $is_biased_choice);
        
        $_SESSION['rl_last_reward'] = $current_reward;
        
        $path = __DIR__ . "/../results_rl/{$schedule_type}/{$schedule_name}/{$model}/";
        if (!file_exists($path)) {
            mkdir($path, 0777, true);
        }
        
        $file_name = $path . $user_id . '.csv';
        if ($t === 0) {
            $header = "trial_number,time,schedule_type,schedule_name,model,is_biased_choice,side_choice,observed_reward,unobserved_reward,biased_reward,unbiased_reward,total_reward\n";
            file_put_contents($file_name, $header);
        }
        
        date_default_timezone_set('Asia/Shanghai');
        $trial_data = $t . ','
            . date("Y-m-d H:i:s") . ','
            . $schedule_type . ','
            . $schedule_name . ','
            . $model . ','
            . strtolower($is_biased_choice) . ','
            . $rl_action . ','
            . $current_reward . ','
            . $current_unobserved_reward . ','
            . $current_biased_reward . ','
            . $current_unbiased_reward . ','
            . $_SESSION['rl_total_reward'] . "\n";
        file_put_contents($file_name, $trial_data, FILE_APPEND);
        
        if ($current_reward) {
            $_SESSION['rl_total_reward']++;
        }
        
        $all_trials[] = [
            'trial' => $t + 1,
            'action' => $rl_action,
            'reward' => $current_reward,
            'is_biased' => $is_biased_choice
        ];
    }
    
    $total_line = "total,,"
        . $schedule_type . ","
        . $schedule_name . ","
        . $model . ",,"
        . $_SESSION['rl_left_count'] . "/" . $_SESSION['rl_right_count'] . ","
        . $_SESSION['rl_total_reward'] . ",,,,\n";
    file_put_contents($file_name, $total_line, FILE_APPEND);
    
    $response['success'] = true;
    $response['data'] = [
        'user_id' => $user_id,
        'model' => $model,
        'schedule_type' => $schedule_type,
        'schedule_name' => $schedule_name,
        'total_trials' => $number_of_trials,
        'total_reward' => $_SESSION['rl_total_reward'],
        'left_count' => $_SESSION['rl_left_count'],
        'right_count' => $_SESSION['rl_right_count'],
        'biased_side' => $_SESSION['rl_biased_side'],
        'csv_path' => "results_rl/{$schedule_type}/{$schedule_name}/{$model}/{$user_id}.csv",
        'trials' => $all_trials
    ];
    echo json_encode($response);
    exit;
}

if ($action === 'status') {
    if (!isset($_SESSION['rl_user_id'])) {
        $response['error'] = 'No active session.';
    } else {
        $response['success'] = true;
        $response['data'] = [
            'user_id' => $_SESSION['rl_user_id'],
            'model' => $_SESSION['rl_model'],
            'schedule_type' => $_SESSION['rl_schedule_type'],
            'schedule_name' => $_SESSION['rl_schedule_name'],
            'current_trial' => $_SESSION['rl_trial_number'],
            'total_trials' => $_SESSION['rl_number_of_trials'],
            'total_reward' => $_SESSION['rl_total_reward'],
            'left_count' => $_SESSION['rl_left_count'],
            'right_count' => $_SESSION['rl_right_count']
        ];
    }
    echo json_encode($response);
    exit;
}

if ($action === 'list_models') {
    $models = ['simple_ql'];
    $models_dir = __DIR__ . "/rl_agents/models/";
    if (is_dir($models_dir)) {
        foreach (glob($models_dir . "*.py") as $file) {
            $models[] = basename($file, '.py');
        }
    }
    $response['success'] = true;
    $response['data'] = ['models' => array_unique($models)];
    echo json_encode($response);
    exit;
}

if ($action === 'list_schedules') {
    $schedules = ['static' => [], 'dynamic' => []];
    
    foreach (glob(__DIR__ . "/sequences/static/*.php") as $file) {
        $schedules['static'][] = basename($file, '.php');
    }
    foreach (glob(__DIR__ . "/custom_sequences/static/*.php") as $file) {
        $schedules['static'][] = basename($file, '.php') . ' (custom)';
    }
    
    foreach (glob(__DIR__ . "/sequences/dynamic/*.py") as $file) {
        $schedules['dynamic'][] = basename($file, '.py');
    }
    foreach (glob(__DIR__ . "/custom_sequences/dynamic/*.py") as $file) {
        $schedules['dynamic'][] = basename($file, '.py') . ' (custom)';
    }
    
    $response['success'] = true;
    $response['data'] = $schedules;
    echo json_encode($response);
    exit;
}

$response['error'] = 'Unknown action. Available: init, run_trial, run_all, status, list_models, list_schedules';
echo json_encode($response);
?>
