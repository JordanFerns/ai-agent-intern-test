"""Entrypoint for Aster & Row Support Agent."""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        from evaluation.eval_runner import run_all_evaluations
        save_dest = sys.argv[2] if len(sys.argv) > 2 else "evaluation/final_results.json"
        run_all_evaluations(save_path=save_dest)
    elif len(sys.argv) > 1 and sys.argv[1] == "web":
        from src.web import app
        print("Starting Aster & Row Web Demo at http://127.0.0.1:5000 ...")
        app.run(host="127.0.0.1", port=5000, debug=False)
    else:
        from src.cli import run_cli
        run_cli()
