"""CLI:  python -m dxaqc run --input <папка|zip> --output <папка> [--format xlsx|csv|both]"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

from . import __version__
from .infer import run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dxaqc", description="Контроль качества DXA-исследований")
    ap.add_argument("--version", action="version", version=f"dxaqc {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="пакетная обработка папки или zip с DICOM")
    r.add_argument("--input", "-i", required=True)
    r.add_argument("--output", "-o", required=True)
    r.add_argument("--format", choices=["xlsx", "csv", "both"], default="both")
    r.add_argument("--overlays", action="store_true", help="записать доп. серию PNG с разметкой нарушений в <output>/overlays/")
    r.add_argument("--mode", choices=["competition", "sensitive"], default=None,
                   help="пороги: competition — максимум F1 (по умолчанию); sensitive — чувствительность ≥ 0,9 по критерию (клиника)")
    args = ap.parse_args(argv)

    if args.mode:
        os.environ["DXAQC_MODE"] = args.mode
    warnings.filterwarnings("ignore")
    # UID организатора длиннее стандарта DICOM — pydicom пишет предупреждение на каждый файл; это шум
    logging.getLogger("pydicom").setLevel(logging.ERROR)
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.output, "run.log"), encoding="utf-8"),
                                  logging.StreamHandler(sys.stderr)])
    try:
        _, s = run(args.input, args.output, fmt=args.format, overlays=args.overlays)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 2
    logging.info("файлов %d | Success %d | Failure %d | уникальных снимков %d | %.2f с",
                 s.files, s.success, s.failure, s.unique_images, s.seconds)
    print(s.xlsx or s.csv)
    return 0          # частичные Failure — не ошибка запуска: они зафиксированы в таблице


if __name__ == "__main__":
    raise SystemExit(main())
