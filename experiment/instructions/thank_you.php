<?php
session_start();
// 计算最终费用：基础5元 + 浮动奖励（每个笑脸0.2元）
$base_fee = 5; // 基础参与费5元
$rewards = isset($_SESSION["total_reward"]) ? intval($_SESSION["total_reward"]) : 0; // 来自 backend.php 的 rewards
$float_fee = $rewards * 0.2; // 每个笑脸0.2元
$final_fee = $base_fee + $float_fee;
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
    <link rel="stylesheet" href="../style/style.css">
    <script>
        function set_payment_code(){
            var code_element = document.getElementById("payment_id");
            // 显示服务器生成的 user_id
            code_element.innerHTML = <?php echo json_encode(isset($_SESSION['user_id'])?strval($_SESSION['user_id']):'——'); ?>;
        }
    </script>
</head>
<body onload="set_payment_code()">

<nav class="navbar navbar-inverse">
    <div class="container-fluid">
        <div class="navbar-header">
            <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#myNavbar">
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
            </button>
            <a class="navbar-brand" href="">重复选择</a>
        </div>
        <div class="collapse navbar-collapse" id="myNavbar">
            <ul class="nav navbar-nav">
                <li class="inactive_nav">(1) 说明</li>
                <li class="inactive_nav">(2) 任务</li>
                <li class="active_nav">(3) 领取报酬</li>
                <li id="filler">(4) 填充项 </li>
            </ul>
            <ul class="nav navbar-nav navbar-right">
            </ul>
        </div>
    </div>
</nav>

<div class="container-fluid text-center">
    <div class="row content">
        <!--sidebar on the left-->
        <div class="col-sm-2 sidenav">
        </div>
        <div class="col-sm-8 text-left">
            <h1>感谢你的参与</h1>
            <hr>
            <!-- 核心文本整合：费用+编号+超链接问卷 -->
            <p style="line-height: 1.8; font-size: 16px;">
                您最终获得的费用是：<strong><?php echo number_format($final_fee, 2); ?>元</strong>（基础参与费5元 + 浮动费用<?php echo number_format($float_fee, 2); ?>元；<?php echo $rewards; ?> 个笑脸，0.2元/笑脸）<br>
                您的编号是：<strong id="user_id_text"><?php echo isset($_SESSION['user_id'])?htmlspecialchars($_SESSION['user_id']):'——'; ?></strong>（请记住您的编号并填入后续问卷中，这将作为您获得费用的凭证）<br>
                请点击下面链接完成问卷，后等待主试发放费用（微信每日加好友有限额，请耐心等待）<br>
                <a href="https://v.wjx.cn/vm/h9vLRjt.aspx#" target="_blank" style="color: #428bca; text-decoration: underline;">https://v.wjx.cn/vm/h9vLRjt.aspx#</a>
            </p>
            <br>
            <h4>您的编号：</h4>
            <h3 id="payment_id" style="color: #d9534f;"></h3>
            <p>
                问卷填写完毕后可以关闭此窗口。
            </p>
        </div>
        <div class="col-sm-2 sidenav">
        </div>
    </div>
</div>

<footer class="container-fluid text-center">
    <p> </p>
</footer>

</body>
</html>