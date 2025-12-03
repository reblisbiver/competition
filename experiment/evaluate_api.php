<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Cache-Control: no-cache');

$response = ['success' => false, 'error' => '', 'data' => []];

$VENV_PYTHON = __DIR__ . '/../.venv/bin/python3';
if (!file_exists($VENV_PYTHON)) {
    $VENV_PYTHON = 'python3';
}

$models = $_GET['models'] ?? 'all';
$runs = intval($_GET['runs'] ?? 5);
$sample = intval($_GET['sample'] ?? 0);
$rulesJson = $_GET['rules'] ?? '';

$runs = max(1, min(10, $runs));
$sample = max(0, min(1500, $sample));

$rules = [];
if ($rulesJson) {
    $decoded = json_decode($rulesJson, true);
    if (is_array($decoded)) {
        $rules = $decoded;
    }
}

$scriptDir = __DIR__ . '/rl_agents/neural_ql';
$evalScript = $scriptDir . '/evaluate_all.py';

if (!file_exists($evalScript)) {
    $response['error'] = 'Evaluation script not found';
    echo json_encode($response);
    exit;
}

$cmd = "cd " . escapeshellarg($scriptDir) . " && " .
       escapeshellarg($VENV_PYTHON) . " " . escapeshellarg($evalScript) .
       " --model " . escapeshellarg($models) .
       " --runs " . escapeshellarg($runs);

if ($sample > 0) {
    $cmd .= " --sample " . escapeshellarg($sample);
}

if (!empty($rules)) {
    $cmd .= " --rules " . escapeshellarg(json_encode($rules));
}

$cmd .= " 2>&1";

$output = [];
$returnCode = 0;

set_time_limit(600);

exec($cmd, $output, $returnCode);

$outputStr = implode("\n", $output);

$resultsFile = $scriptDir . '/evaluation_results/all_models_results.json';

if (file_exists($resultsFile)) {
    $results = json_decode(file_get_contents($resultsFile), true);
    
    if ($results) {
        $response['success'] = true;
        $response['data'] = $results;
        $response['output'] = $outputStr;
    } else {
        $response['error'] = 'Failed to parse results';
        $response['output'] = $outputStr;
    }
} else {
    $response['error'] = 'Results file not generated';
    $response['output'] = $outputStr;
    $response['return_code'] = $returnCode;
}

echo json_encode($response);
