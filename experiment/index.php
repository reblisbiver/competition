<?php
// Pass URL parameters to welcome.html
$redirect_url = "instructions/welcome.html";

if (isset($_GET['schedule_type']) || isset($_GET['schedule_name'])) {
    $redirect_url .= "?";
    $params = [];
    if (isset($_GET['schedule_type'])) {
        $params[] = "schedule_type=" . urlencode($_GET['schedule_type']);
    }
    if (isset($_GET['schedule_name'])) {
        $params[] = "schedule_name=" . urlencode($_GET['schedule_name']);
    }
    $redirect_url .= implode("&", $params);
}

header("Location: " . $redirect_url);
exit();
?>
