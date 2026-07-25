from ml_runtime_bench.cli import build_parser


def test_cli_parses_suite() -> None:
    parser = build_parser()
    args = parser.parse_args(["suite", "--config", "configs/smoke.json", "--output", "results/x"])
    assert args.command == "suite"
