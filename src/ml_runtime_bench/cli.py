from __future__ import annotations

import argparse
from pathlib import Path

from ml_runtime_bench.benchmark import run_suite
from ml_runtime_bench.config import ModelConfig, load_suite_config
from ml_runtime_bench.diagnostics import run_graph_break_diagnostics
from ml_runtime_bench.dynamic_shapes import run_dynamic_shape_experiment
from ml_runtime_bench.plotting import plot_results
from ml_runtime_bench.profiling import run_profile
from ml_runtime_bench.reporting import write_suite_outputs
from ml_runtime_bench.triton_kernels import benchmark_triton_rmsnorm


def _add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)


def _model_config(args: argparse.Namespace) -> ModelConfig:
    config = ModelConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
    )
    config.validate()
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-runtime-bench",
        description="PyTorch inference runtime and compiler benchmark suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    suite = subparsers.add_parser("suite", help="run a configured benchmark suite")
    suite.add_argument("--config", required=True, type=Path)
    suite.add_argument("--output", required=True, type=Path)
    suite.add_argument("--strict", action="store_true")

    profile = subparsers.add_parser("profile", help="generate a PyTorch profiler trace")
    profile.add_argument("--mode", choices=["eager", "compile"], default="eager")
    profile.add_argument("--device", default="auto")
    profile.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    profile.add_argument("--batch-size", type=int, default=2)
    profile.add_argument("--sequence-length", type=int, default=128)
    profile.add_argument("--warmup-iterations", type=int, default=3)
    profile.add_argument("--profile-iterations", type=int, default=5)
    profile.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="default",
    )
    profile.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    profile.add_argument("--output", required=True, type=Path)
    _add_common_model_arguments(profile)

    diagnose = subparsers.add_parser("diagnose", help="analyze a TorchDynamo graph break")
    diagnose.add_argument("--device", default="auto")
    diagnose.add_argument("--output", required=True, type=Path)

    dynamic = subparsers.add_parser(
        "dynamic-shapes", help="measure shape specialization and repeat-call cost"
    )
    dynamic.add_argument("--device", default="auto")
    dynamic.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    dynamic.add_argument("--batch-size", type=int, default=1)
    dynamic.add_argument("--lengths", nargs="+", type=int, required=True)
    dynamic.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    dynamic.add_argument("--output", required=True, type=Path)
    _add_common_model_arguments(dynamic)

    triton_parser = subparsers.add_parser(
        "triton-rmsnorm", help="benchmark the optional fused Triton RMSNorm kernel"
    )
    triton_parser.add_argument("--hidden-sizes", nargs="+", type=int, required=True)
    triton_parser.add_argument("--rows", type=int, default=2048)
    triton_parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    triton_parser.add_argument("--warmup", type=int, default=20)
    triton_parser.add_argument("--iterations", type=int, default=100)
    triton_parser.add_argument("--output", required=True, type=Path)

    plot = subparsers.add_parser("plot", help="create latency and throughput charts")
    plot.add_argument("--results", required=True, type=Path)
    plot.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "suite":
        suite_config = load_suite_config(args.config)
        payload = run_suite(suite_config, args.output, strict=args.strict)
        write_suite_outputs(payload, args.output)
        print(f"Wrote benchmark outputs to {args.output}")
        return

    if args.command == "profile":
        run_profile(
            model_config=_model_config(args),
            mode=args.mode,
            device_name=args.device,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            warmup_iterations=args.warmup_iterations,
            profile_iterations=args.profile_iterations,
            output_dir=args.output,
            compile_mode=args.compile_mode,
            dynamic=args.dynamic,
        )
        print(f"Wrote profiler outputs to {args.output}")
        return

    if args.command == "diagnose":
        run_graph_break_diagnostics(output_dir=args.output, device_name=args.device)
        print(f"Wrote graph-break outputs to {args.output}")
        return

    if args.command == "dynamic-shapes":
        model_config = _model_config(args)
        run_dynamic_shape_experiment(
            output_dir=args.output,
            lengths=args.lengths,
            batch_size=args.batch_size,
            device_name=args.device,
            dtype_name=args.dtype,
            dynamic=args.dynamic,
            model_config=model_config,
        )
        print(f"Wrote dynamic-shape outputs to {args.output}")
        return

    if args.command == "triton-rmsnorm":
        benchmark_triton_rmsnorm(
            hidden_sizes=args.hidden_sizes,
            rows=args.rows,
            dtype_name=args.dtype,
            output_dir=args.output,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"Wrote Triton outputs to {args.output}")
        return

    if args.command == "plot":
        plot_results(args.results, args.output)
        print(f"Wrote plots to {args.output}")
        return

    parser.error(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
