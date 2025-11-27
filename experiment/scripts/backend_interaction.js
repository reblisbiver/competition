function get_current_trial_rewards(side_chosen) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            rewards_arrived(xmlHttp.responseText);
    };
    xmlHttp.open(
        "GET",
        "scripts/backend.php?side=" +
            side_chosen +
            "&RT=" +
            RT +
            "&NUMBER_OF_TRIALS=" +
            NUMBER_OF_TRIALS,
        true,
    ); // true for asynchronous
    xmlHttp.send();
}

function rewards_arrived(reward) {
    show_feedback(Number(reward));
    if (trial_number == NUMBER_OF_TRIALS) {
        // 新增：调用完整数据保存接口
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open("POST", "scripts/save_complete_data.php", true);
        xmlHttp.setRequestHeader("Content-Type", "application/json");
        xmlHttp.send(JSON.stringify({})); // 无需额外数据，通过SESSION获取积累的试次数据

        // 跳转感谢页面（原逻辑保留）
        go_to_goodbye_page();
    }
}
