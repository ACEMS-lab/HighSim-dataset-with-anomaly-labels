"""
pipeline.py —— 统一入口：原始 HighSim 数据 → 特征、训练、检测、基线对比。

用法：
    python pipeline.py --workdir runs/latest --stage all
    python pipeline.py --workdir runs/latest --stage detect      # 只重跑检测（复用已训练模型）

阶段（按顺序，all = 全部）：
    data       Data.py           平滑速度/加速度、TTC、数据质量标记、GT 事件
    features   model.py          → phi_all.csv（φ1–φ4）
    train      model.py          → gruae_model.keras, norm_params.npz
    detect     model.py          → GMM 分类 UCL、帧/事件级检测与评估
    baselines  baselines.py      → 基线/消融对比 experiment_results*.csv

所有输出写入 --workdir。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STAGES = ["data", "features", "train", "detect", "baselines"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=os.path.join(HERE, "runs", "latest"),
                    help="输出目录（默认 runs/latest）")
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "highsim_download", "HighSim_Data_20260601.csv"),
                    help="原始 HighSim 数据（无表头 CSV）。默认新版 20260601（含货车 Class=3）；"
                         "旧版 'HighSim Data.csv' 也可用，列顺序自动识别")
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--from-stage", default=None, choices=STAGES,
                    help="从该阶段开始一直跑到最后，例如 --from-stage train")
    ap.add_argument("--epochs", type=int, default=200,
                    help="训练轮数上限；实际由早停 (patience=10, 恢复最优权重) 决定")
    args = ap.parse_args()

    raw = os.path.abspath(args.raw)
    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)
    print(f"[pipeline] workdir = {os.getcwd()}")

    if args.from_stage:
        todo = STAGES[STAGES.index(args.from_stage):]
    elif args.stage == "all":
        todo = STAGES
    else:
        todo = [args.stage]

    for st in todo:
        print(f"\n[pipeline] ===== stage: {st} =====", flush=True)
        if st == "data":
            import Data
            Data.main(raw)
        elif st in ("features", "train", "detect"):
            import model
            {"features": model.stage_features,
             "train":    lambda: model.stage_train(epochs=args.epochs),
             "detect":   model.stage_detect}[st]()
        elif st == "baselines":
            import baselines
            baselines.main()
    print("\n[pipeline] done.")


if __name__ == "__main__":
    main()
