<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');

$response = ['success' => false, 'error' => '', 'data' => []];

$VENV_PYTHON = __DIR__ . '/../.venv/bin/python3';
if (!file_exists($VENV_PYTHON)) {
    $VENV_PYTHON = 'python3';
}

$action = $_GET['action'] ?? 'run_trial';
$rl_model = $_GET['model'] ?? 'simple_ql';
$schedule_type = strtoupper($_GET['schedule_type'] ?? 'STATIC');
$schedule_name = $_GET['schedule_name'] ?? 'random_0';
$number_of_trials = intval($_GET['trials'] ?? 100);
$session_id = $_GET['session_id'] ?? null;

$SESSIONS_DIR = __DIR__ . '/../rl_sessions/';
if (!file_exists($SESSIONS_DIR)) {
    mkdir($SESSIONS_DIR, 0777, true);
}

function validate_session_id($session_id) {
    if (!$session_id || !is_string($session_id)) {
        return false;
    }
    if (!preg_match('/^[A-Za-z0-9_\-]+$/', $session_id)) {
        return false;
    }
    if (strlen($session_id) > 50) {
        return false;
    }
    return true;
}

function load_session($session_id) {
    global $SESSIONS_DIR;
    if (!validate_session_id($session_id)) {
        return null;
    }
    $file = $SESSIONS_DIR . $session_id . '.json';
    if (file_exists($file)) {
        return json_decode(file_get_contents($file), true);
    }
    return null;
}

function save_session($session_id, $data) {
    global $SESSIONS_DIR;
    if (!validate_session_id($session_id)) {
        return false;
    }
    $file = $SESSIONS_DIR . $session_id . '.json';
    return file_put_contents($file, json_encode($data)) !== false;
}

function delete_session($session_id) {
    global $SESSIONS_DIR;
    if (!validate_session_id($session_id)) {
        return false;
    }
    $file = $SESSIONS_DIR . $session_id . '.json';
    if (file_exists($file)) {
        return unlink($file);
    }
    return true;
}

function generate_session_id() {
    return 'rl_' . time() . '_' . bin2hex(random_bytes(8));
}

if ($action === 'init') {
    $new_session_id = generate_session_id();
    
    $session_data = [
        'user_id' => $new_session_id,
        'model' => $rl_model,
        'schedule_type' => $schedule_type,
        'schedule_name' => $schedule_name,
        'number_of_trials' => $number_of_trials,
        'trial_number' => 0,
        'total_reward' => 0,
        'last_action' => null,
        'last_reward' => null,
        'bias_rewards' => [],
        'unbias_rewards' => [],
        'is_bias_choice' => [],
        'left_count' => 0,
        'right_count' => 0,
        'biased_side' => (rand(1,2) == 1) ? 'LEFT' : 'RIGHT',
        'created_at' => date('Y-m-d H:i:s')
    ];
    $session_data['unbiased_side'] = ($session_data['biased_side'] === 'LEFT') ? 'RIGHT' : 'LEFT';
    
    save_session($new_session_id, $session_data);
    
    $response['success'] = true;
    $response['data'] = [
        'session_id' => $new_session_id,
        'user_id' => $new_session_id,
        'model' => $rl_model,
        'schedule_type' => $schedule_type,
        'schedule_name' => $schedule_name,
        'total_trials' => $number_of_trials,
        'biased_side' => $session_data['biased_side'],
        'message' => 'Session initialized. Use session_id in subsequent calls.'
    ];
    echo json_encode($response);
    exit;
}

if ($action === 'run_trial') {
    if (!$session_id) {
        $response['error'] = 'Missing session_id parameter. Call ?action=init first to get a session_id.';
        echo json_encode($response);
        exit;
    }
    
    $sess = load_session($session_id);
    if (!$sess) {
        $response['error'] = 'Invalid or expired session_id.';
        echo json_encode($response);
        exit;
    }
    
    if ($sess['trial_number'] >= $sess['number_of_trials']) {
        $response['success'] = true;
        $response['data'] = [
            'status' => 'completed',
            'session_id' => $session_id,
            'total_reward' => $sess['total_reward'],
            'total_trials' => $sess['number_of_trials'],
            'left_count' => $sess['left_count'],
            'right_count' => $sess['right_count'],
            'message' => 'All trials completed!'
        ];
        echo json_encode($response);
        exit;
    }
    
    $user_id = $sess['user_id'];
    $model = $sess['model'];
    $last_action = $sess['last_action'] ?? 'None';
    $last_reward = $sess['last_reward'] ?? '0';
    
    $model_script = __DIR__ . "/rl_agents/models/{$model}.py";
    if (!file_exists($model_script)) {
        $model_script = __DIR__ . "/rl_agents/simple_ql.py";
    }
    
    global $VENV_PYTHON;
    $command = "{$VENV_PYTHON} {$model_script} " . escapeshellarg($user_id) . " " . escapeshellarg($last_action) . " " . escapeshellarg($last_reward) . " 2>&1";
    $rl_action = trim(exec($command, $output, $return_code));
    
    if ($return_code !== 0 || !in_array($rl_action, ['LEFT', 'RIGHT'])) {
        $response['error'] = "RL model error: " . implode("\n", $output);
        echo json_encode($response);
        exit;
    }
    
    $sess['last_action'] = $rl_action;
    
    if ($rl_action === 'LEFT') {
        $sess['left_count']++;
    } else {
        $sess['right_count']++;
    }
    
    $is_biased_side = ($sess['biased_side'] === $rl_action);
    $is_biased_choice = $is_biased_side ? 'True' : 'False';
    
    $current_biased_reward = $current_unbiased_reward = 0;
    $schedule_type = $sess['schedule_type'];
    $schedule_name = $sess['schedule_name'];
    
    $custom_static_path = __DIR__ . "/custom_sequences/static/{$schedule_name}.php";
    $custom_dynamic_path = __DIR__ . "/custom_sequences/dynamic/{$schedule_name}.py";
    $default_static_path = __DIR__ . "/sequences/static/{$schedule_name}.php";
    
    if ($schedule_type === 'DYNAMIC') {
        if (file_exists($custom_dynamic_path)) {
            $script_path = $custom_dynamic_path;
        } else {
            $script_path = __DIR__ . "/sequences/dynamic/{$schedule_name}.py";
        }
        
        global $VENV_PYTHON;
        $run_python_command = "{$VENV_PYTHON} {$script_path} "
            . escapeshellarg(json_encode($sess['bias_rewards'])) . ' '
            . escapeshellarg(json_encode($sess['unbias_rewards'])) . ' '
            . escapeshellarg(json_encode($sess['is_bias_choice'])) . ' '
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
        $trial_idx = $sess['trial_number'];
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
    
    $sess['bias_rewards'][] = $current_biased_reward;
    $sess['unbias_rewards'][] = $current_unbiased_reward;
    $sess['is_bias_choice'][] = $is_biased_choice;
    
    $sess['last_reward'] = $current_reward;
    
    if ($current_reward) {
        $sess['total_reward']++;
    }
    
    $sess['trial_number']++;
    
    $path = __DIR__ . "/../results_rl/{$schedule_type}/{$schedule_name}/{$model}/";
    if (!file_exists($path)) {
        mkdir($path, 0777, true);
    }
    
    $file_name = $path . $user_id . '.csv';
    if ($sess['trial_number'] === 1) {
        $header = "trial_number,time,schedule_type,schedule_name,model,is_biased_choice,side_choice,observed_reward,unobserved_reward,biased_reward,unbiased_reward,total_reward\n";
        file_put_contents($file_name, $header);
    }
    
    date_default_timezone_set('Asia/Shanghai');
    $trial_data = ($sess['trial_number'] - 1) . ','
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
        . $sess['total_reward'] . "\n";
    file_put_contents($file_name, $trial_data, FILE_APPEND);
    
    $is_last_trial = ($sess['trial_number'] >= $sess['number_of_trials']);
    
    if ($is_last_trial) {
        $total_line = "total,,"
            . $schedule_type . ","
            . $schedule_name . ","
            . $model . ",,"
            . $sess['left_count'] . "/" . $sess['right_count'] . ","
            . $sess['total_reward'] . ",,,,\n";
        file_put_contents($file_name, $total_line, FILE_APPEND);
        delete_session($session_id);
    } else {
        save_session($session_id, $sess);
    }
    
    $response['success'] = true;
    $response['data'] = [
        'session_id' => $session_id,
        'trial' => $sess['trial_number'],
        'total_trials' => $sess['number_of_trials'],
        'action' => $rl_action,
        'reward' => $current_reward,
        'total_reward' => $sess['total_reward'],
        'is_biased_choice' => $is_biased_choice,
        'status' => $is_last_trial ? 'completed' : 'in_progress',
        'left_count' => $sess['left_count'],
        'right_count' => $sess['right_count']
    ];
    echo json_encode($response);
    exit;
}

if ($action === 'run_all') {
    $new_session_id = generate_session_id();
    
    $sess = [
        'user_id' => $new_session_id,
        'model' => $rl_model,
        'schedule_type' => $schedule_type,
        'schedule_name' => $schedule_name,
        'number_of_trials' => $number_of_trials,
        'trial_number' => 0,
        'total_reward' => 0,
        'last_action' => null,
        'last_reward' => null,
        'bias_rewards' => [],
        'unbias_rewards' => [],
        'is_bias_choice' => [],
        'left_count' => 0,
        'right_count' => 0,
        'biased_side' => (rand(1,2) == 1) ? 'LEFT' : 'RIGHT'
    ];
    $sess['unbiased_side'] = ($sess['biased_side'] === 'LEFT') ? 'RIGHT' : 'LEFT';
    
    $user_id = $sess['user_id'];
    $model = $sess['model'];
    $all_trials = [];
    
    $path = __DIR__ . "/../results_rl/{$schedule_type}/{$schedule_name}/{$model}/";
    if (!file_exists($path)) {
        mkdir($path, 0777, true);
    }
    $file_name = $path . $user_id . '.csv';
    $header = "trial_number,time,schedule_type,schedule_name,model,is_biased_choice,side_choice,observed_reward,unobserved_reward,biased_reward,unbiased_reward,total_reward\n";
    file_put_contents($file_name, $header);
    
    for ($t = 0; $t < $number_of_trials; $t++) {
        $last_action = $sess['last_action'] ?? 'None';
        $last_reward = $sess['last_reward'] ?? '0';
        
        $model_script = __DIR__ . "/rl_agents/models/{$model}.py";
        if (!file_exists($model_script)) {
            $model_script = __DIR__ . "/rl_agents/simple_ql.py";
        }
        
        global $VENV_PYTHON;
        $command = "{$VENV_PYTHON} {$model_script} " . escapeshellarg($user_id) . " " . escapeshellarg($last_action) . " " . escapeshellarg($last_reward) . " 2>&1";
        $rl_action = trim(exec($command, $output, $return_code));
        $output = [];
        
        if ($return_code !== 0 || !in_array($rl_action, ['LEFT', 'RIGHT'])) {
            $response['error'] = "RL model error at trial $t: action='$rl_action'";
            echo json_encode($response);
            exit;
        }
        
        $sess['last_action'] = $rl_action;
        
        if ($rl_action === 'LEFT') {
            $sess['left_count']++;
        } else {
            $sess['right_count']++;
        }
        
        $is_biased_side = ($sess['biased_side'] === $rl_action);
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
            
            global $VENV_PYTHON;
            $run_python_command = "{$VENV_PYTHON} {$script_path} "
                . escapeshellarg(json_encode($sess['bias_rewards'])) . ' '
                . escapeshellarg(json_encode($sess['unbias_rewards'])) . ' '
                . escapeshellarg(json_encode($sess['is_bias_choice'])) . ' '
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
        
        $sess['bias_rewards'][] = $current_biased_reward;
        $sess['unbias_rewards'][] = $current_unbiased_reward;
        $sess['is_bias_choice'][] = $is_biased_choice;
        
        $sess['last_reward'] = $current_reward;
        
        if ($current_reward) {
            $sess['total_reward']++;
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
            . $sess['total_reward'] . "\n";
        file_put_contents($file_name, $trial_data, FILE_APPEND);
        
        $all_trials[] = [
            'trial' => $t + 1,
            'action' => $rl_action,
            'reward' => $current_reward,
            'is_biased' => $is_biased_choice,
            'total_reward' => $sess['total_reward']
        ];
    }
    
    $total_line = "total,,"
        . $schedule_type . ","
        . $schedule_name . ","
        . $model . ",,"
        . $sess['left_count'] . "/" . $sess['right_count'] . ","
        . $sess['total_reward'] . ",,,,\n";
    file_put_contents($file_name, $total_line, FILE_APPEND);
    
    $response['success'] = true;
    $response['data'] = [
        'session_id' => $new_session_id,
        'user_id' => $user_id,
        'model' => $model,
        'schedule_type' => $schedule_type,
        'schedule_name' => $schedule_name,
        'total_trials' => $number_of_trials,
        'total_reward' => $sess['total_reward'],
        'left_count' => $sess['left_count'],
        'right_count' => $sess['right_count'],
        'biased_side' => $sess['biased_side'],
        'csv_path' => "results_rl/{$schedule_type}/{$schedule_name}/{$model}/{$user_id}.csv",
        'trials' => $all_trials
    ];
    echo json_encode($response);
    exit;
}

if ($action === 'status') {
    if (!$session_id) {
        $response['error'] = 'Missing session_id parameter.';
        echo json_encode($response);
        exit;
    }
    
    $sess = load_session($session_id);
    if (!$sess) {
        $response['error'] = 'Invalid or expired session_id.';
    } else {
        $response['success'] = true;
        $response['data'] = [
            'session_id' => $session_id,
            'user_id' => $sess['user_id'],
            'model' => $sess['model'],
            'schedule_type' => $sess['schedule_type'],
            'schedule_name' => $sess['schedule_name'],
            'current_trial' => $sess['trial_number'],
            'total_trials' => $sess['number_of_trials'],
            'total_reward' => $sess['total_reward'],
            'left_count' => $sess['left_count'],
            'right_count' => $sess['right_count'],
            'biased_side' => $sess['biased_side']
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
            $name = basename($file, '.py');
            if ($name !== 'base_model') {
                $models[] = $name;
            }
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
