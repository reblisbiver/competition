<?php
session_start(); // 开启会话
echo "数据已收齐，感谢您的参与";
// // 1. 健康检查逻辑（保留你要求的代码）
// if ($_SERVER['REQUEST_URI'] === '/' || $_SERVER['REQUEST_URI'] === '/index.php') {
//     // Check if this is a health check (no browser headers)
//     if (!isset($_SERVER['HTTP_ACCEPT']) || strpos($_SERVER['HTTP_ACCEPT'], 'text/html') === false) {
//         header('Content-Type: text/plain');
//         http_response_code(200);
//         echo "OK";
//         exit;
//     }
// }

// // 2. 处理同意/不同意逻辑（点击同意跳转到instructions/welcome.html）
// if ($_SERVER['REQUEST_METHOD'] === 'POST') {
//     if (isset($_POST['agree'])) {
//         // 同意后跳转至指定页面
//         header("Location: instructions/welcome.html"); 
//         exit;
//     } elseif (isset($_POST['disagree'])) {
//         echo "<script>alert('您已选择退出实验，感谢关注！'); window.close();</script>";
//         exit;
//     }
// }

// 3. 显示知情同意书（默认GET请求/刷新时显示）
?>
<!-- 知情同意书页面（完整迁移过来） -->
<!-- <!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实验知情同意书</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
        .container { background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; color: #333; }
        .section { margin: 20px 0; }
        .btn-group { text-align: center; margin-top: 30px; }
        button { padding: 10px 30px; margin: 0 10px; border: none; border-radius: 4px; cursor: pointer; }
        .agree-btn { background: #4CAF50; color: white; }
        .disagree-btn { background: #f44336; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h2>知情同意书</h2>
        <p>欢迎参加实验！本实验旨在研究人们的行为选择，您的参与将帮助我们深入了解人类行为的科学原则。以下是关于实验的说明：</p>

        <div class="section">
            <h3>实验过程</h3>
            <p>本实验中，您将通过电脑端完成100道二选一的题目，在每一道题目中选择不同的选项可能会给您带来不同的收益，您需要做的是尽可能提高自己的收益。我们向您承诺，本实验涉及的情景和设置全部是真实的。如果您有任何关于实验的疑问，请按照下面联系方式联系我们。</p>
        </div>

        <div class="section">
            <h3>实验时长及报酬</h3>
            <p>本次实验时长约8分钟，实验参与报酬组成为，基础参与费5元，浮动费用0-10元，全部试次完成后方能获得费用。</p>
        </div>

        <div class="section">
            <h3>研究风险</h3>
            <p>本实验对接受试验的人员无已知的健康风险。虽然和其他行为实验一样，本实验有极小的信息泄露的风险，但研究人员会妥善保存所有研究数据，绝不会用于除研究以外的任何其他用途。</p>
        </div>

        <div class="section">
            <h3>参与的自愿性与机密性</h3>
            <p>如果您决定参加本项研究，您的个人资料及在实验中的数据均将被严格保密，不会透露给研究小组以外的任何成员。您的个人信息、实验数据将以实验编号而非您的姓名加以标识，这些数据将和您的姓名与联系方式分开保存。本研究的结果可能会发表在国际学术刊物上，但与您个人有关的资料都不会被公开，也不会出现在任何出版物上。作为接受试验者，您可随时了解与本实验有关的信息资料和研究进展。</p>
            <p>在实验期间及实验结束后的一年内，您可以随时选择退出本研究，退出无需任何理由。一旦您选择退出，我们将立即中止您的实验，并删除与您相关的所有数据。</p>
        </div>

        <div class="section">
            <h3>联系方式</h3>
            <p>周钰坪</p>
            <p>Email: zhouyuping@stu.pku.edu.cn</p>
        </div>

        <div class="section">
            <h3>知情同意声明</h3>
            <p>“我已被告知本研究的背景、目的、步骤、风险及获益情况。我有足够的时间和机会进行提问和考虑。我已阅读此知情同意书，并且同意参加本研究。我知道我可以在研究期间的任何时候无需任何理由退出本研究。”</p>
        </div>

        <form method="post" class="btn-group">
            <button type="submit" name="agree" class="agree-btn">同意且继续</button>
            <button type="submit" name="disagree" class="disagree-btn">不同意并退出</button>
        </form>
    </div>
</body>
</html> -->