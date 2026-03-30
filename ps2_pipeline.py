"""
PS2 - Autonomous Structural Intelligence System
Complete pipeline: Stage 1 (OpenCV) → Stage 2 (Classification) → Stage 4 (Materials) → Stage 5 (LLM)
Stage 3 (Three.js 3D) is in index.html — open that in your browser.

SETUP (run once in your terminal):
    pip install opencv-python shapely numpy anthropic

USAGE:
    python ps2_pipeline.py floorplan.png
"""

import cv2
import numpy as np
import json
import sys
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"   # replace with your key
USE_LLM = False   # set True once you have an API key

# ─── MATERIAL DATABASE (from problem statement) ───────────────────────────────
MATERIALS = [
    {"name": "AAC Blocks",      "cost": "Low",      "strength": "Medium",    "durability": "High",      "best_use": "Partition walls"},
    {"name": "Red Brick",       "cost": "Medium",   "strength": "High",      "durability": "Medium",    "best_use": "Load-bearing walls"},
    {"name": "RCC",             "cost": "High",     "strength": "Very High", "durability": "Very High", "best_use": "Columns, slabs"},
    {"name": "Steel Frame",     "cost": "High",     "strength": "Very High", "durability": "Very High", "best_use": "Long spans (>5m)"},
    {"name": "Hollow Concrete", "cost": "Low",      "strength": "Medium",    "durability": "Medium",    "best_use": "Non-structural walls"},
    {"name": "Fly Ash Brick",   "cost": "Low",      "strength": "High",      "durability": "High",      "best_use": "General walling"},
    {"name": "Precast Concrete","cost": "Med-High", "strength": "High",      "durability": "Very High", "best_use": "Structural walls, slabs"},
]

COST_MAP      = {"Low": 3, "Low-Med": 2.5, "Med-High": 1.5, "Medium": 2, "High": 1}
STRENGTH_MAP  = {"Medium": 2, "High": 3, "Very High": 4}
DURABILITY_MAP = {"Medium": 2, "High": 3, "Very High": 4}

# ─── STAGE 1: FLOOR PLAN PARSING ──────────────────────────────────────────────
def parse_floor_plan(image_path):
    print(f"\n[Stage 1] Parsing floor plan: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print("  Image not found — using synthetic test floor plan")
        img = create_test_floor_plan()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
        threshold=80, minLineLength=40, maxLineGap=10)

    walls = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            walls.append({"x1": int(x1), "y1": int(y1),
                          "x2": int(x2), "y2": int(y2),
                          "length_px": round(length, 1)})

    print(f"  Detected {len(walls)} wall segments")
    return img, walls


def create_test_floor_plan():
    """Creates a synthetic floor plan for testing when no image is provided."""
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (650, 450), (0, 0, 0), 4)
    cv2.line(img, (300, 50), (300, 300), (0, 0, 0), 3)
    cv2.line(img, (50, 250), (300, 250), (0, 0, 0), 3)
    cv2.line(img, (300, 200), (650, 200), (0, 0, 0), 3)
    cv2.line(img, (480, 200), (480, 450), (0, 0, 0), 3)
    return img


# ─── STAGE 2: GEOMETRY RECONSTRUCTION ─────────────────────────────────────────
def classify_walls(walls, px_per_meter=50):
    print("\n[Stage 2] Classifying walls...")

    if not walls:
        return walls

    all_x = [w["x1"] for w in walls] + [w["x2"] for w in walls]
    all_y = [w["y1"] for w in walls] + [w["y2"] for w in walls]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    MARGIN = 20

    for w in walls:
        is_outer = (
            min(w["x1"], w["x2"]) <= min_x + MARGIN or
            max(w["x1"], w["x2"]) >= max_x - MARGIN or
            min(w["y1"], w["y2"]) <= min_y + MARGIN or
            max(w["y1"], w["y2"]) >= max_y - MARGIN
        )
        w["wall_type"] = "load_bearing" if is_outer else "partition"
        w["length_m"] = round(w["length_px"] / px_per_meter, 2)
        print(f"  ({w['x1']},{w['y1']})→({w['x2']},{w['y2']})  "
              f"{w['wall_type']:15s}  {w['length_m']}m")

    return walls


# ─── STAGE 4: MATERIAL ANALYSIS ───────────────────────────────────────────────
def material_score(mat, wall_type):
    if wall_type == "load_bearing":
        cw, sw, dw = 0.25, 0.50, 0.25   # strength matters most
    else:
        cw, sw, dw = 0.50, 0.25, 0.25   # cost matters most for partition

    c = COST_MAP.get(mat["cost"], 2)
    s = STRENGTH_MAP.get(mat["strength"], 2)
    d = DURABILITY_MAP.get(mat["durability"], 2)
    return round(c * cw + s * sw + d * dw, 2)


def analyse_materials(walls):
    print("\n[Stage 4] Analysing materials...")
    report = []

    for i, wall in enumerate(walls):
        scored = sorted(
            [{"material": m["name"],
              "score": material_score(m, wall["wall_type"]),
              "cost": m["cost"],
              "strength": m["strength"],
              "best_use": m["best_use"]}
             for m in MATERIALS],
            key=lambda x: x["score"], reverse=True
        )[:3]

        report.append({"wall_id": i + 1, "wall": wall, "recommendations": scored})

        print(f"\n  Wall {i+1} ({wall['wall_type']}, {wall['length_m']}m):")
        for rank, rec in enumerate(scored, 1):
            print(f"    #{rank} {rec['material']:20s} score={rec['score']}  "
                  f"cost={rec['cost']}, strength={rec['strength']}")

    return report


# ─── STAGE 5: EXPLAINABILITY ───────────────────────────────────────────────────
def explain_with_llm(wall, recs):
    """Call Claude API to generate explanation. Falls back to template if no key."""
    if not USE_LLM or ANTHROPIC_API_KEY == "YOUR_API_KEY_HERE":
        return explain_template(wall, recs)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = f"""
You are a structural engineering assistant. Given the wall data below, write
2-3 sentences explaining why the top material is recommended. Be specific —
mention the span length, wall type, and cost-vs-strength tradeoff score.

Wall: {wall['wall_type'].replace('_', '-')} wall, span = {wall['length_m']}m
Top recommendation: {recs[0]['material']} (score: {recs[0]['score']})
  Cost: {recs[0]['cost']}, Strength: {recs[0]['strength']}
Alternative: {recs[1]['material']} (score: {recs[1]['score']})
  Cost: {recs[1]['cost']}, Strength: {recs[1]['strength']}

Write the explanation now:
"""
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"  LLM error: {e} — using template")
        return explain_template(wall, recs)


def explain_template(wall, recs):
    """Fallback template explanation (no API needed)."""
    top, alt = recs[0], recs[1]
    return (
        f"{top['material']} is recommended for this {wall['wall_type'].replace('_','-')} "
        f"wall (span: {wall['length_m']}m) because its {top['strength'].lower()} strength "
        f"rating suits the structural demands at this span length, with a tradeoff score "
        f"of {top['score']} reflecting its {top['cost'].lower()} cost profile. "
        f"{alt['material']} (score {alt['score']}) is the alternative, offering "
        f"{alt['strength'].lower()} strength at {alt['cost'].lower()} cost."
    )


def generate_explanations(report):
    print("\n[Stage 5] Generating explanations...")
    for entry in report:
        explanation = explain_with_llm(entry["wall"], entry["recommendations"])
        entry["explanation"] = explanation
        print(f"\n  Wall {entry['wall_id']}:")
        print(f"  {explanation}")
    return report


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "floorplan.png"

    # Run pipeline
    img, walls = parse_floor_plan(image_path)
    walls = classify_walls(walls)
    material_report = analyse_materials(walls)
    final_report = generate_explanations(material_report)

    # Save output
    output_file = "ps2_output.json"
    with open(output_file, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Pipeline complete! Output saved to {output_file}")
    print(f"Next: open index.html in your browser for the 3D model.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
