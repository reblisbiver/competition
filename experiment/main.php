<?php
session_start();
//**********************************************************************************************************************
// This is the main file which runs the experiment, it also initiates and defines the parameters of the game
//**********************************************************************************************************************

// Assign worker ID as a persistent incremental integer (1,2,3...)
$counter_file = __DIR__ . '/user_id_counter.txt';
$new_id = null;
// Ensure counter file exists
if (!file_exists($counter_file)) {
    file_put_contents($counter_file, "0");
}
$fp = @fopen($counter_file, 'c+');
if ($fp) {
    // exclusive lock
    if (flock($fp, LOCK_EX)) {
        // read current value
        $contents = stream_get_contents($fp);
        $contents = trim($contents);
        $last = ($contents === '') ? 0 : intval($contents);
        $new_id = $last + 1;
        // write new value
        ftruncate($fp, 0);
        rewind($fp);
        fwrite($fp, (string)$new_id);
        fflush($fp);
        flock($fp, LOCK_UN);
    }
    fclose($fp);
}
if ($new_id === null) {
    // fallback to timestamp-based id if file ops fail
    $new_id = intval(time());
}
$_SESSION['user_id'] = strval($new_id);

$biased_side = 'biased_side';
$unbiased_side = 'unbiased_side';

// Rewards and choices loggers
$_SESSION['is_bias_choice'] = array();
$_SESSION['bias_rewards'] = array();
$_SESSION['unbias_rewards'] = array();

// Choose reward assignment type
$TYPE_DYNAMIC = "DYNAMIC";
$TYPE_STATIC = "STATIC";

// Helper: build schedule list from sequences folders
$sequences_dir = __DIR__ . '/sequences';
$dynamic_dir = $sequences_dir . '/dynamic';
$static_dir = $sequences_dir . '/static';
$schedule_list = array();
if (is_dir($dynamic_dir)) {
    foreach (glob($dynamic_dir . '/*.py') as $file) {
        $schedule_list[] = basename($file, '.py');
    }
}
if (is_dir($static_dir)) {
    foreach (glob($static_dir . '/*.php') as $file) {
        $schedule_list[] = basename($file, '.php');
    }
}
// remove duplicates
$schedule_list = array_values(array_unique($schedule_list));

// persistent counts file for schedules (minimal implementation)
$counts_file = __DIR__ . '/schedules_counts.json';
$counts = array();
if (file_exists($counts_file)) {
    $raw = file_get_contents($counts_file);
    $decoded = json_decode($raw, true);
    if (is_array($decoded)) {
        $counts = $decoded;
    }
}
// ensure all schedules have an entry
foreach ($schedule_list as $name) {
    if (!isset($counts[$name])) $counts[$name] = 0;
}
// write back if we added entries
file_put_contents($counts_file, json_encode($counts, JSON_PRETTY_PRINT));

// Read from URL parameters or use defaults for schedule_type (may be overridden below)
$_SESSION['schedule_type'] = isset($_GET['schedule_type']) ? strtoupper($_GET['schedule_type']) : $TYPE_DYNAMIC;
if ($_SESSION['schedule_type'] !== $TYPE_DYNAMIC && $_SESSION['schedule_type'] !== $TYPE_STATIC) {
    $_SESSION['schedule_type'] = $TYPE_DYNAMIC;
}

// Choose specific sequence (from the "sequences" folder)
if (isset($_GET['schedule_name']) && strlen($_GET['schedule_name'])) {
    $_SESSION['schedule_name'] = $_GET['schedule_name'];
} else {
    // pick least-called schedule (minimal tie-break: first found)
    $min_count = null;
    $candidate = null;
    foreach ($schedule_list as $name) {
        $c = isset($counts[$name]) ? intval($counts[$name]) : 0;
        if ($min_count === null || $c < $min_count) {
            $min_count = $c;
            $candidate = $name;
        }
    }
    // fallback
    if ($candidate === null) $candidate = (count($schedule_list) ? $schedule_list[0] : 'bilstm_dynamic');
    $_SESSION['schedule_name'] = $candidate;
}

// determine schedule_type from existence if not explicitly provided
if (!isset($_GET['schedule_type']) || $_GET['schedule_type']=='') {
    $name = $_SESSION['schedule_name'];
    if (file_exists($dynamic_dir . '/' . $name . '.py')) {
        $_SESSION['schedule_type'] = $TYPE_DYNAMIC;
    } elseif (file_exists($static_dir . '/' . $name . '.php')) {
        $_SESSION['schedule_type'] = $TYPE_STATIC;
    }
}

// Choose Right/Left buttons as biased/anti-biased side
if (rand(1,2)==1){
    $_SESSION[$biased_side] = "LEFT";
    $_SESSION[$unbiased_side] = "RIGHT";
}
else{
    $_SESSION[$biased_side] = "RIGHT";
    $_SESSION[$unbiased_side] = "LEFT";
}

// Init game
$_SESSION['trial_number'] = 0;
$_SESSION["total_reward"] = 0;

?>
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <title>重复选择</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
    <link rel="stylesheet" href="style/style.css">
    <script src="scripts/backend_interaction.js"></script>
    <script src="scripts/code.js"></script>

</head>
<body onload="init_page()">

<nav class="navbar navbar-inverse">
    <div class="container-fluid">
        <div class="navbar-header">
            <span class="navbar-brand">重复选择</span>
        </div>
        <div class="collapse navbar-collapse" id="myNavbar">
            <ul class="nav navbar-nav">
                <li class="inactive_nav">(1) 说明</li>
                <li class="active_nav">(2) 任务</li>
                <li class="inactive_nav">(3) 领取报酬</li>
                <li id="filler">(4) 填充项 </li>
            </ul>
            <ul class="nav navbar-nav navbar-right">
                <li>
                    <p class="navbar-text" style="margin-right:10px; color:#ddd;">Schedule: <?php echo htmlspecialchars($_SESSION['schedule_name']); ?></p>
                </li>
            </ul>
        </div>
    </div>
</nav>

<div class="container-fluid text-center">
    <div class="row content">
        <div class="col-sm-2 sidenav justify_text" id="please_choose_text">
            <br>
            <p>
                请在两个按钮之间选择。
            </p>
            <p>
                你的目标是收集尽可能多的笑脸表情。
            </p>
        </div>
        <div class="col-sm-8 text-center centered_content" id="button_table">
            <table style="width:80%">
                <tr>
                    <th></th>
                    <th></th>
                    <th></th>
                </tr>

                <tr> <!--Here the image is displayed-->
                    <th>
                        <button type="button" class="btn btn-default button_left" id="button_left" onclick="left_button_clicked()"></button>
                    </th>
                    <!--Seperator-->
                    <th id="feedback_placeholder"> </th>
                    <th>
                        <button type="button" class="btn btn-default button_right" id="button_right" onclick="right_button_clicked()"></button>
                    </th>
                </tr>

                <tr >
                </tr>
                <tr class="counter">
                    <th> </th>
                    <th>
                        <p>已收集总奖励：</p><div id="total_reward">0</div>
                    </th>
                    <th> </th>
            </table>
        </div>
        <div class="col-sm-2 sidenav" id="reward_holder">

        </div>
    </div>
</div>

<footer class="container-fluid text-center" id="text_left">
    <!--<p>Footer Text</p>-->
    <p>
        <div id="trials_left"></div> 剩余试验次数。
    </p>
</footer>

</body>
</html>