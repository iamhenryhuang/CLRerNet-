import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_PAIR_RE = re.compile(r"\b(loss(?:_[a-zA-Z0-9]+)?)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
STEP_RE = re.compile(r"Epoch\(train\)\s*\[(\d+)\]\[\s*(\d+)\s*/\s*(\d+)\]")


def parse_log(log_path: Path):
    data = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Epoch(train)" not in line:
                continue

            step_m = STEP_RE.search(line)
            if step_m:
                epoch = int(step_m.group(1))
                it = int(step_m.group(2))
                total_it = int(step_m.group(3))
                x = (epoch - 1) * total_it + it
            else:
                x = None

            pairs = LOSS_PAIR_RE.findall(line)
            if not pairs:
                continue

            for name, val in pairs:
                if name not in data:
                    data[name] = {"x": [], "y": []}
                data[name]["x"].append(x if x is not None else len(data[name]["y"]) + 1)
                data[name]["y"].append(float(val))
    return data


def save_plots(loss_data, out_dir: Path, dpi: int = 140):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for loss_name in sorted(loss_data.keys()):
        xs = loss_data[loss_name]["x"]
        ys = loss_data[loss_name]["y"]
        if not ys:
            continue

        plt.figure(figsize=(8, 4.5))
        plt.plot(xs, ys, linewidth=1.2)
        plt.title(loss_name)
        plt.xlabel("global_iter")
        plt.ylabel(loss_name)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_file = out_dir / f"{loss_name}.png"
        plt.savefig(out_file, dpi=dpi)
        plt.close()
        saved.append(out_file)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--out-dir", default="", help="Output folder for loss plots")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = (Path.cwd() / log_path).resolve()

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (log_path.parent / "loss_plots")
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    loss_data = parse_log(log_path)
    if not loss_data:
        raise RuntimeError("No train loss records found in log.")

    saved = save_plots(loss_data, out_dir)

    print(f"Log: {log_path}")
    print(f"Output dir: {out_dir}")
    print(f"Loss curves saved: {len(saved)}")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
