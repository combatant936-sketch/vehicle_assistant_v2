FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked
COPY download.py ./
RUN uv run python download.py

COPY . .

EXPOSE 8000
CMD ["uvicorn", "project.app:app", "--host", "0.0.0.0", "--port", "8000"]