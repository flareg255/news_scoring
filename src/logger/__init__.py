"""
ロギングユーティリティ

使用例:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("処理開始")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
_initialized = False


def setup(level: int = logging.DEBUG) -> None:
    """
    アプリケーション全体のロギングを初期化する。
    main.py の先頭で1度だけ呼ぶこと。
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # コンソール出力（INFO以上）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(console)

    # ファイル出力（DEBUG以上、全量）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(f"ログ出力先: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """モジュール名を渡してロガーを取得する。"""
    return logging.getLogger(name)
