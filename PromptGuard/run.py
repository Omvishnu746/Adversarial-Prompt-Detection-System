"""
PromptGuard - Application Launcher

Starts the FastAPI server using uvicorn.

Usage:
    python run.py
    python run.py --host 0.0.0.0 --port 8080 --no-reload
"""

import sys
import argparse
import uvicorn

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptGuard API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload (use in production)",
    )
    args = parser.parse_args()

    print(f"\n[PromptGuard v0 - Phase 1]")
    print(f"  Listening on  : http://{args.host}:{args.port}")
    print(f"  Docs          : http://{args.host}:{args.port}/docs")
    print(f"  Health check  : http://{args.host}:{args.port}/health\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()
