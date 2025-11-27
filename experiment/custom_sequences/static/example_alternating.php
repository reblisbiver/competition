<?php
/**
 * Example Custom Static Sequence: Alternating rewards
 * 
 * Rewards alternate between sides every 10 trials.
 */

$biased_rewards = [];
$unbiased_rewards = [];

for ($i = 0; $i < 100; $i++) {
    $block = floor($i / 10) % 2;
    if ($block == 0) {
        $biased_rewards[] = 1;
        $unbiased_rewards[] = 0;
    } else {
        $biased_rewards[] = 0;
        $unbiased_rewards[] = 1;
    }
}
?>
