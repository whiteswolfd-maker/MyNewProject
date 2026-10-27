"""训练（新启动场，单半径或多半径共用一条逻辑）。

.. code-block:: bash

    # 程序诊断：10 步，不主张科研验收
    python -m spherical_blast_pinn.cli.train --config configs/experiments/startup_diagnostic.yaml \
        --diagnostic --max-steps 10

    # 从检查点续训（完整训练动态状态）
    python -m spherical_blast_pinn.cli.train --config ... --resume outputs/run/last.pt

    # 只加载参数、重新开始优化
    python -m spherical_blast_pinn.cli.train --config ... --init-from outputs/run/last.pt

正式的完整日程是 20000 次更新；``--max-steps`` 只用于小规模联检。诊断运行有
明确的步数上限，且不会把未获科研验收的启动场当成正式结果。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ..training.schedule import SCHEDULE, TOTAL_UPDATES
from ..training.startup_cases import CaseSetupError
from ..training.trainer import (
    DEFAULT_DIAGNOSTIC_MAX_UPDATES,
    RUN_COMPLETED,
    RUN_CRITERIA_FAILED,
    RUN_DIAGNOSTIC,
    RUN_INCOMPLETE,
    RUN_STATUS_MEANING,
    Trainer,
    TrainingError,
)
from ._common import (
    SetupError,
    add_common_arguments,
    build_context,
    configure_stdout,
    write_json,
)

EXIT_CODES: dict[str, int] = {
    RUN_COMPLETED: 0,
    RUN_INCOMPLETE: 1,
    RUN_CRITERIA_FAILED: 2,
    RUN_DIAGNOSTIC: 0,
}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except (SetupError, TrainingError, CaseSetupError) as exc:
        print(f"\n** {exc} **\n", file=sys.stderr)
        return 6


def _run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--resume", default=None,
        help="从检查点恢复完整训练状态（模型/优化器/采样窗口/RNG/阶段）",
    )
    parser.add_argument(
        "--init-from", default=None,
        help="只加载模型参数，重新开始优化（不恢复步号、优化器或采样状态）",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="总更新次数上限（绝对步号，不是新增步数）",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--lbfgs", type=int, default=None,
        help="可选的 L-BFGS 精修迭代次数（固定采样点，最多 500）",
    )
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="诊断运行：允许未验收启动场，步数有上限，不主张任何验收结论",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    configure_stdout()

    from ._common import load_config_from_args

    config = load_config_from_args(args)
    if args.output_dir:
        config = config.set("training.output_dir", args.output_dir)
    if args.resume and args.init_from:
        print("\n** --resume 与 --init-from 只能二选一 **\n", file=sys.stderr)
        return 7

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
                f"\n** 训练未开始：training.max_updates 必须是 0 或正整数 **\n",
                file=sys.stderr,
            )
            return 7
    cap = int(config.get(
        "training.diagnostic_max_updates", DEFAULT_DIAGNOSTIC_MAX_UPDATES
    ))
    if args.diagnostic and budget is None:
        budget = cap
    if args.diagnostic and budget is not None and budget > cap:
        print(
            f"\n** 诊断运行上限是 {cap} 步；收到 {budget} 步。"
            "正式训练需要科研验收通过的启动场。**\n",
            file=sys.stderr,
        )
        return 7

    context = build_context(
        config,
        verbose=args.verbose,
        seed=args.seed,
        diagnostic=bool(args.diagnostic),
        device=config.get("training.device", "cpu"),
    )

    if context.data_contract is not None and not context.data_contract.ready:
        print("\n** 训练未开始：数据契约没有放行 **\n", file=sys.stderr)
        for reason in context.data_contract.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 4

    isolation = getattr(context, "validation_isolation", None)
    if isolation is not None and not isolation.ok:
        print("\n** 训练未开始：验证隔离被破坏 **\n", file=sys.stderr)
        for reason in isolation.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 5

    output_dir = config.path_of("training.output_dir", default=Path("outputs/run"))
    trainer = Trainer(
        cases=context.training_cases,
        model=context.model,
        schedule=SCHEDULE,
        seed=int(args.seed if args.seed is not None else 0),
        output_dir=output_dir,
        log_every=int(config.get("training.log_every", 50)),
        checkpoint_every=int(config.get("training.checkpoint_every", 500)),
        check_every=int(config.get("training.check_every", 500)),
        gradient_clip=float(config.get("training.gradient_clip", 1.0)),
        tolerances=dict(config.section("acceptance")),
        diagnostic=bool(args.diagnostic),
        diagnostic_max_updates=cap,
        config=config,
    )
    if args.resume:
        info = trainer.load_checkpoint(args.resume)
        print(f"resumed from {info['path']} at step {info['step']}")
    elif args.init_from:
        trainer.load_weights(args.init_from)
        print(f"loaded model parameters from {args.init_from}; optimiser state is fresh")

    print(
        f"training {len(context.training_cases)} startup case(s) "
        f"({', '.join(context.training_cases[i].case_id for i in range(len(context.training_cases)))})"
    )
    for case in context.training_cases:
        print(
            f"  {case.case_id}: R0={case.charge_radius:g} m  "
            f"epsilon={case.field.epsilon:.6e} s  tau_end={case.tau_end:g} s  "
            f"p_star={case.contact_pressure:.6e} Pa"
            + ("  [synthetic supervision]" if case.data is not None and case.data.synthetic else "")
            + ("  [real supervision]" if case.data is not None and not case.data.synthetic else "")
        )
    if args.diagnostic:
        print(
            f"** DIAGNOSTIC run: capped at {cap} updates; this makes no scientific "
            "acceptance claim. **"
        )

    result = trainer.run(
        max_updates=budget, verbose=True, diagnostic=bool(args.diagnostic)
    )
    if args.lbfgs:
        result = trainer.refine(iterations=args.lbfgs, verbose=True)

    write_json(output_dir / "train_result.json", result.as_dict())
    print(
        f"\nupdates = {result.updates}  final loss = {result.final_loss:.6e} "
        f"(measured at step {result.final_loss_step})"
    )
    print(f"status = {result.status}: {RUN_STATUS_MEANING.get(result.status, '')}")
    print(f"written: {output_dir}")
    if result.status == RUN_INCOMPLETE:
        print(
            f"** 这是截断运行，不是完整训练：完整日程是 {TOTAL_UPDATES} 次更新。**"
        )
    elif result.status == RUN_CRITERIA_FAILED:
        print(
            "** 预算跑满了，但最终阶段的判据没有通过：这不是一次正式成功。",
            file=sys.stderr,
        )
        for row in (result.final_stage_verdict or {}).get("criteria", []):
            if not row.get("passed"):
                print(
                    f"  - {row['metric']} = {row['value']:.6g} "
                    f"(needs {row['sense']} {row['limit']:.6g})",
                    file=sys.stderr,
                )
    elif result.status == RUN_DIAGNOSTIC:
        print(
            "** 这是一次诊断运行：不主张任何科研验收结论，也不是正式训练完成。",
            file=sys.stderr,
        )
    return EXIT_CODES[result.status]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
