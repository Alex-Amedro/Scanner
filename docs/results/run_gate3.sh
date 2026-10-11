#!/bin/bash
# Runs every measurement of gates 2 and 3 (same houses and seeds as gate 1). Usage: bash docs/results/run_gate3.sh  (from the repository root)
cd "$(dirname "$0")/../../src"
R=../docs/results
python pose_eval.py --levels 1 2 3 4 --noise low medium high --pose-source slam_lc --seeds 9000 9020 --procs 8 --out $R/gate3_loop_closure.csv > $R/gate3_log.txt 2>&1
python pose_eval.py --levels 1 2 3 4 --noise low medium high --pose-source slam --seeds 9000 9020 --procs 8 --out $R/gate2_scan_matching.csv > $R/gate2_log.txt 2>&1
python pose_eval.py --levels 2 3 4 --noise extreme --pose-source odom --seeds 9000 9020 --procs 8 --out $R/extreme_odom.csv > $R/extreme_odom_log.txt 2>&1
python pose_eval.py --levels 2 3 4 --noise extreme --pose-source slam --seeds 9000 9020 --procs 8 --out $R/extreme_slam.csv > $R/extreme_slam_log.txt 2>&1
python pose_eval.py --levels 2 3 4 --noise extreme --pose-source slam_lc --seeds 9000 9020 --procs 8 --out $R/extreme_slam_lc.csv > $R/extreme_slam_lc_log.txt 2>&1
echo FINI > $R/gate3_done.txt
