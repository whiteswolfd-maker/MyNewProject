"""训练（提示词 §13）。

.. code-block:: bash

    python -m spherical_blast_pinn.cli.train --config configs/experiments/smoke.yaml
    python -m spherical_blast_pinn.cli.train --config configs/experiments/full.yaml --resume outputs/full/last.pt

训练**不会被自动发起**。完整日程是 20000 次更新；``--max-steps`` 只用来做小规模
联合检查，用它跑出来的东西不是训练完成。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ..data.compatibility import CompatibilityError
from ..training.schedule import SCHEDULE, TOTAL_UPDATES
from ..training.trainer import (
    RUN_COMPLETED,
    RUN_CRITERIA_FAILED,
    RUN_DIAGNOSTIC,
    RUN_INCOMPLETE,
    RUN_STATUS_MEANING,
    Trainer,
    TrainingError,
)
from ._common import add_common_arguments, build_context, configure_stdout, write_json

#: 退出码。**四档终态各自可区分**：以前只有 0/1，"跑完了但判据没过"和"截断"
#: 拿到同一个非零码，调用方分不清哪一种。
EXIT_CODES: dict[str, int] = {
    RUN_COMPLETED: 0,
    RUN_INCOMPLETE: 1,
    RUN_CRITERIA_FAILED: 2,
    # 诊断运行不主张验收结论，但它也不是失败——退出码正常，状态写在清单里。
    RUN_DIAGNOSTIC: 0,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument("--resume", default=None, help="从检查点恢复")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="只跑前若干次更新，用于小规模联合检查，不是训练完成")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lbfgs", type=int, default=None,
                        help="可选的 L-BFGS 精修迭代次数（最多 500，固定采样点）")
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="诊断运行：不主张任何验收结论，状态记作 diagnostic",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    configure_stdout()

    from ._common import load_config_from_args

    config = load_config_from_args(args)
    if args.output_dir:
        config = config.set("training.output_dir", args.output_dir)

    # 预算的解析必须在**昂贵的上下文构造之前**，且 ``None`` 与 ``0`` 必须分开。
    #
    # 以前这一步写的是 ``int(max_updates) if max_updates else None``：``0`` 是假值，
    # 于是"跑零步"被解释成"跑完整日程"——一条用来做冒烟检查的命令会安静地启动
    # 一次完整训练。语义在这里定死：``--max-steps 0`` 表示**零次更新**，运行会立刻
    # 结束并判为 ``incomplete``；不想要这个行为就不要传这个参数。
    if args.max_steps is not None:
        if int(args.max_steps) < 0:
            print(
                f"\n** 训练未开始：--max-steps 必须是 0 或正整数，收到 {args.max_steps} **\n",
                file=sys.stderr,
            )
            return 7
        config = config.set("training.max_updates", int(args.max_steps))
    budget = config.get("training.max_updates", None)
    if budget is not None:
        budget = int(budget)
        if budget < 0:
            print(
                f"\n** 训练未开始：training.max_updates 必须是 0 或正整数，"
                f"收到 {budget} **\n",
                file=sys.stderr,
            )
            return 7

    context = build_context(config, verbose=args.verbose, seed=args.seed)
    if context.model is None:
        raise TrainingError("no model was built; check the configuration")

    # §15：正式训练入口必须**同时**检查初场、模型适用性、数据契约与验证隔离。
    # 初场与物理由 ``build_reference`` → ``assert_reference_accepted`` 拦（缺任何
    # 一项都会抛 ``ReferenceNotAccepted``）；数据这一侧在这里拦。
    #
    # 数据未就绪**不是**可以靠改配置绕过的开关——它意味着监督标签本身不可信，
    # 那样跑出来的任何数字都没有意义。所以这里是硬失败，不是警告。
    if context.data_contract is not None and not context.data_contract.ready:
        print("\n** 训练未开始：数据契约没有放行 **\n", file=sys.stderr)
        for reason in context.data_contract.reasons:
            print(f"  - {reason}", file=sys.stderr)
        print(
            "\n数据契约就绪需要重新提取（见 data/contract.py 与 §15）。"
            "这条不是数值验收失败，也不是可以用配置放宽的门槛。",
            file=sys.stderr,
        )
        return 4

    # §16：最终验证半径（500 mm）不得参与训练、初值标定、归一化极值、早停或选模。
    # 这条以前只靠配置里的一个字符串，没有任何代码读它——把那一行改成 train
    # 什么都不会报错，而报表上仍然写着"盲测半径已隔离"。
    isolation = getattr(context, "validation_isolation", None)
    if isolation is not None and not isolation.ok:
        print("\n** 训练未开始：验证隔离被破坏 **\n", file=sys.stderr)
        for reason in isolation.reasons:
            print(f"  - {reason}", file=sys.stderr)
        print(
            "\n500 mm 只能做最终独立验证（§16）。把它的 split 改回训练会让这次运行的"
            "每一个数字都失去独立验证的意义。",
            file=sys.stderr,
        )
        return 5

    output_dir = config.path_of("training.output_dir", default=Path("outputs/run"))
    try:
        trainer = Trainer(
            config=config,
            cases=context.training_cases,
            reference=context.reference,
            model=context.model,
            scales_by_radius=context.scales,
            time_scale=context.time_scale,
            schedule=SCHEDULE,
            sampling=context.sampling,
            seed=int(args.seed if args.seed is not None else 0),
            device=config.get("training.device", "cpu"),
            output_dir=output_dir,
            log_every=int(config.get("training.log_every", 50)),
            checkpoint_every=int(config.get("training.checkpoint_every", 500)),
            check_every=int(config.get("training.check_every", 500)),
            gradient_clip=float(config.get("training.gradient_clip", 1.0)),
            r_min_fraction=float(config.get("training.r_min_fraction", 1.0e-2)),
            compatibility=context.compatibility,
            compatibility_report=context.compatibility_report,
            tolerances=dict(config.section("acceptance")),
        )
    except CompatibilityError as exc:
        print("\n** 训练未开始：数据兼容性闸门没有放行 **\n", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 3
    if args.resume:
        trainer.load(args.resume)

    # ``None`` 与 ``0`` 分开：``None`` 是完整日程，``0`` 是零次更新。
    print(
        f"training {len(context.training_cases)} radii "
        f"({', '.join(c.case_id for c in context.training_cases)})"
        f" for {budget if budget is not None else TOTAL_UPDATES} updates"
    )
    print(f"hat T_* = {context.time_scale:.6f}  (frozen from training radii only)")
    print(f"contact pressure scale Pc_* = {trainer.contact_pressure:.6e} Pa")
    print(f"data compatibility gate: {trainer.gate.mode}"
          f" (data_compatibility = {'unverified' if trainer.gate.unverified else 'verified'})")
    if trainer.gate.unverified:
        print(f"** {trainer.gate.message}")
    future = trainer.run(max_updates=budget, verbose=True, diagnostic=args.diagnostic)
    if args.lbfgs:
        future = trainer.refine(iterations=args.lbfgs, verbose=True)
    result = future

    write_json(output_dir / "train_result.json", result.as_dict())
    print(
        f"\nupdates = {result.updates}"
        f"  final loss = {result.final_loss:.6e} (measured at step {result.final_loss_step})"
    )
    print(f"status = {result.status}: {RUN_STATUS_MEANING.get(result.status, '')}")
    print(f"written: {output_dir}")
    if result.status == RUN_INCOMPLETE:
        print(
            "** 这是截断运行，不是完整训练：完整日程是 "
            f"{TOTAL_UPDATES} 次更新。它不主张任何验收结论。"
        )
    elif result.status == RUN_CRITERIA_FAILED:
        print(
            "** 预算跑满了，但最终阶段的判据**没有通过**：这不是一次正式成功。",
            file=sys.stderr,
        )
        verdict = result.final_stage_verdict or {}
        for row in verdict.get("criteria", []):
            if not row.get("passed"):
                print(
                    f"  - {row['metric']} = {row['value']:.6g} "
                    f"(needs {row['sense']} {row['limit']:.6g})",
                    file=sys.stderr,
                )
        if verdict.get("reason"):
            print(f"  - {verdict['reason']}", file=sys.stderr)
    elif result.status == RUN_DIAGNOSTIC:
        print(
            "** 这是一次诊断运行：它不主张任何验收结论，也不是正式训练完成。",
            file=sys.stderr,
        )
    return EXIT_CODES[result.status]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
