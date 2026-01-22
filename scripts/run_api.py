from __future__ import annotations

from pathlib import Path

import uvicorn

from fakenews.serving.api import create_app


def main():
    app = create_app(model_dir=Path("artifacts/models/v1"))
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
