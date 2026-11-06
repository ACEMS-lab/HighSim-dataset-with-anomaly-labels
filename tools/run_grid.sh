#!/bin/sh
# 清洗强度 × 控制限 网格：等 v9 跑完后依次执行（避免抢 CPU）
cd "$(dirname "$0")/.."
AB=$(pwd)
while ! grep -q "pipeline\] done\|Traceback" runs/v9/pipeline_log.txt; do sleep 60; done
mkdir -p runs/grid/base
cp runs/v9/HighSim_* runs/v9/accel_threshold_curves.* runs/grid/base/
python pipeline.py --workdir runs/grid/base --stage features > runs/grid/base/log.txt 2>&1 || exit 1
for C in none 3 2; do
  D=runs/grid/c$C; mkdir -p $D
  cp runs/grid/base/HighSim_* runs/grid/base/phi_all.csv runs/grid/base/phi_zmax.json $D/
  CLEAN_Z=$C python pipeline.py --workdir $D --stage train  > $D/log.txt 2>&1 || exit 1
  CLEAN_Z=$C python pipeline.py --workdir $D --stage detect >> $D/log.txt 2>&1 || exit 1
  (cd $D && python $AB/tools/ucl_sweep.py > sweep_log.txt 2>&1)
done
echo GRID_DONE > runs/grid/done.txt
