<?php
// Health check - return 200 for root path
if ($_SERVER['REQUEST_URI'] === '/' || $_SERVER['REQUEST_URI'] === '/index.php') {
    // Check if this is a health check (no browser headers)
    if (!isset($_SERVER['HTTP_ACCEPT']) || strpos($_SERVER['HTTP_ACCEPT'], 'text/html') === false) {
        header('Content-Type: text/plain');
        http_response_code(200);
        echo "OK";
        exit;
    }
}
// Normal redirect to main.php for browsers
header('Location: main.php');
exit;
