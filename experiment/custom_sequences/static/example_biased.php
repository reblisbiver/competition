<?php
/**
 * Example Custom Static Sequence: Biased toward one side
 * 
 * This sequence gives 70% reward probability to the biased side
 * and 30% reward probability to the unbiased side.
 * 
 * How to create your own:
 * 1. Create arrays $biased_rewards and $unbiased_rewards
 * 2. Each array must have exactly 100 elements (0 or 1)
 * 3. Save as .php file in this folder
 * 4. Call API with schedule_name=your_file_name (without .php)
 */

$biased_rewards = [];
$unbiased_rewards = [];

for ($i = 0; $i < 100; $i++) {
    $biased_rewards[] = (rand(1, 100) <= 70) ? 1 : 0;
    $unbiased_rewards[] = (rand(1, 100) <= 30) ? 1 : 0;
}
?>
