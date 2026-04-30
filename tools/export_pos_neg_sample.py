import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_lines_txt(lines_path: Path, w: int, h: int):
    lanes = []
    if not lines_path.exists():
        return lanes
    with lines_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            pts = []
            for i in range(0, len(vals), 2):
                if i + 1 >= len(vals):
                    break
                x = int(np.clip(vals[i], 0, w - 1))
                y = int(np.clip(vals[i + 1], 0, h - 1))
                pts.append([x, y])
            if len(pts) >= 2:
                lanes.append(np.array(pts, dtype=np.int32))
    return lanes


def make_lane_mask(h: int, w: int, lanes, thickness: int = 8):
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in lanes:
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=int(thickness))
    return mask


def make_perspective_lane_mask(
    h: int,
    w: int,
    lanes,
    min_thickness: int = 1,
    max_thickness: int = 4,
):
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in lanes:
        if len(pts) < 2:
            continue
        for p0, p1 in zip(pts[:-1], pts[1:]):
            y_mid = float(p0[1] + p1[1]) * 0.5
            y_ratio = np.clip(y_mid / max(h - 1, 1), 0.0, 1.0)
            thickness = int(round(min_thickness + (max_thickness - min_thickness) * y_ratio))
            thickness = max(min_thickness, min(max_thickness, thickness))
            cv2.line(mask, tuple(p0), tuple(p1), 1, thickness=thickness)
    return mask


def apply_lane_enhance_positive(
    img_bgr_u8: np.ndarray,
    mask_hw: np.ndarray,
    lane_brighten: float = 0.28,
):
    """Mirror CLRerNet._apply_lane_enhance_positive for a single image.

    Current logic in detector:
    - lane_mask = mask.clamp(0, 1)
    - positive = (x_unit + lane_brighten * lane_mask).clamp(0, 1)
    """
    x = img_bgr_u8.astype(np.float32) / 255.0
    lane_mask = np.clip(mask_hw.astype(np.float32), 0.0, 1.0)

    if float(lane_mask.sum()) <= 0:
        positive = x
    else:
        positive = np.clip(x + float(lane_brighten) * lane_mask[:, :, None], 0.0, 1.0)

    pos_u8 = np.clip(np.round(positive * 255.0), 0, 255).astype(np.uint8)
    lane_mask_u8 = np.clip(np.round(lane_mask * 255.0), 0, 255).astype(np.uint8)
    return pos_u8, lane_mask_u8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="dataset/culane/driver_100_30frame/05251330_0404.MP4/01380.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--lines",
        default="",
        help="Path to .lines.txt (default: same basename as image)",
    )
    parser.add_argument(
        "--out-dir",
        default="debug_views/pos_neg_samples",
        help="Output directory",
    )
    parser.add_argument(
        "--lane-brighten",
        type=float,
        default=0.28,
        help="Lane-only brighten strength (matches CLRerNet positive branch)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    img_path = (root / args.image).resolve() if not os.path.isabs(args.image) else Path(args.image)
    if args.lines:
        lines_path = (root / args.lines).resolve() if not os.path.isabs(args.lines) else Path(args.lines)
    else:
        lines_path = img_path.with_suffix(".lines.txt")

    out_dir = (root / args.out_dir).resolve() if not os.path.isabs(args.out_dir) else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    lanes = parse_lines_txt(lines_path, w, h)
    # Match detector behavior:
    # - positive branch uses perspective thickness (1~4)
    # - negative branch uses line_thickness=8
    mask_pos = make_perspective_lane_mask(h, w, lanes, min_thickness=1, max_thickness=4)
    mask_neg = make_lane_mask(h, w, lanes, thickness=8)

    # Positive sample (matches CLRerNet._apply_lane_enhance_positive)
    pos, lane_alpha = apply_lane_enhance_positive(
        img,
        mask_pos,
        lane_brighten=args.lane_brighten,
    )

    # Negative sample (matches detector: dilate + Telea inpaint)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_dil = cv2.dilate(mask_neg, kernel, iterations=1)
    neg = cv2.inpaint(img, mask_dil, 7, cv2.INPAINT_TELEA)

    cv2.imwrite(str(out_dir / "anchor.png"), img)
    cv2.imwrite(str(out_dir / "positive_lane_enhance.png"), pos)
    cv2.imwrite(str(out_dir / "negative_inpaint.png"), neg)
    cv2.imwrite(str(out_dir / "lane_mask_positive_t1.png"), (mask_pos * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "lane_mask_negative_t8.png"), (mask_neg * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "lane_mask_dilated.png"), (mask_dil * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "lane_alpha_soft.png"), lane_alpha)

    print(f"Saved to: {out_dir}")
    print(f"Image: {img_path}")
    print(f"Lines: {lines_path}")
    print(f"Lanes found: {len(lanes)}")
    print(f"lane_brighten: {args.lane_brighten}")


if __name__ == "__main__":
    main()
