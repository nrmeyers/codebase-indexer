"""Entrypoint: python main.py  (or uvicorn app.main:app)."""
from __future__ import annotations

import uvicorn

from app.config import settings
from app.main import app


def main() -> None:
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)


if __name__ == "__main__":
    main()
