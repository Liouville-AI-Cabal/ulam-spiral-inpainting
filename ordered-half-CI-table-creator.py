# Build a workbook with numeric value tables (3 decimals) + green-yellow-red coloring,
# and separate sheets holding the ±CI half-width tables (same layout/order).
import pandas as pd, re, os
import numpy as np

REPORT_DIR = "/home/jen/Desktop/ulam-spiral-inpainting2/visual_samples"
out_path = "model_report_colored_tables.xlsx"

patterns = {
    "title": re.compile(r"FINAL RESULTS for (.+)"),
    "overall_accuracy": re.compile(r"Overall Accuracy/Micro F1:\s*([\d.\-eE]+)"),
    "micro_ci": re.compile(r"Micro CI\s*:\s*\[([\d.\-eE]+)\s*-\s*([\d.\-eE]+)\]"),
    "macro_f1": re.compile(r"Macro F1\s*:\s*([\d.\-eE]+)"),
    "macro_ci": re.compile(r"Macro CI\s*:\s*\[([\d.\-eE]+)\s*-\s*([\d.\-eE]+)\]"),
    "class1_precision": re.compile(r"Class 1 Precision\s*:\s*([\d.]+)\s*\(CI 95%:\s*\[([\d.\-eE]+)\s*-\s*([\d.\-eE]+)\]\)"),
    "class1_recall": re.compile(r"Class 1 Recall\s*:\s*([\d.]+)\s*\(CI 95%:\s*\[([\d.\-eE]+)\s*-\s*([\d.\-eE]+)\]\)"),
    "class1_f1": re.compile(r"Class 1 F1\s*:\s*([\d.]+)\s*\(CI 95%:\s*\[([\d.\-eE]+)\s*-\s*([\d.\-eE]+)\]\)"),
    "class0_precision": re.compile(r"Class 0 Precision\s*:\s*([\d.]+)\s*\(CI 95%:\s*\[([\d.\-eE]+)\s*-\s*([\d.\-eE]+)\]\)"),
    "class0_recall": re.compile(r"Class 0 Recall\s*:\s*([\d.\-eE]+)\s*\(CI 95%:\s*\[([\d.\-eE]+)\s*-\s*([\d.\-eE]+)\]\)"),
    "class0_f1": re.compile(r"Class 0 F1\s*:\s*([\d.\-eE]+)\s*\(CI 95%:\s*\[([\d.\-eE]+)\s*-\s*([\d.\-eE]+)\]\)"),
}

def parse_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    row = {"run_name": os.path.splitext(os.path.basename(path))[0]}
    m = patterns["title"].search(text)
    if m:
        row["run_name"] = m.group(1).strip()
    for key in ["overall_accuracy","macro_f1"]:
        m = patterns[key].search(text)
        if m:
            row[key] = float(m.group(1))
    m = patterns["micro_ci"].search(text)
    if m:
        row["overall_accuracy_ci_lo"], row["overall_accuracy_ci_hi"] = float(m.group(1)), float(m.group(2))
    m = patterns["macro_ci"].search(text)
    if m:
        row["macro_f1_ci_lo"], row["macro_f1_ci_hi"] = float(m.group(1)), float(m.group(2))
    for key in ["class1_precision","class1_recall","class1_f1","class0_precision","class0_recall","class0_f1"]:
        m = patterns[key].search(text)
        if m:
            row[key] = float(m.group(1))
            row[key+"_ci_lo"], row[key+"_ci_hi"] = float(m.group(2)), float(m.group(3))
    return row

rows = []
for fn in os.listdir(REPORT_DIR):
    if fn.endswith(".txt"):
        try:
            rows.append(parse_file(os.path.join(REPORT_DIR, fn)))
        except Exception:
            pass

df = pd.DataFrame(rows)

def parse_train_test_labels(text: str):
    t = str(text).replace(",", "")
    m = re.search(r"_(\d+)[mM]?\s*_on\s*_(\d+)[mM]?", t)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"trained[_\-]?(\d+)[mM]?.*(?:eval|on)[_\-]?(\d+)[mM]?", t)
    if m: return (int(m.group(1)), int(m.group(2)))
    nums = re.findall(r"(\d{2,})", t)
    if len(nums) >= 2: return (int(nums[0]), int(nums[1]))
    return (text, text)

df["trained_on"], df["tested_on"] = zip(*df["run_name"].map(parse_train_test_labels))

metrics = [
    ("overall_accuracy", "Overall Accuracy (Micro F1)"),
    ("macro_f1", "Macro F1"),
    ("class1_precision", "Class 1 Precision"),
    ("class1_recall", "Class 1 Recall"),
    ("class1_f1", "Class 1 F1"),
    ("class0_precision", "Class 0 Precision"),
    ("class0_recall", "Class 0 Recall"),
    ("class0_f1", "Class 0 F1"),
]

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    # Keep raw table for reference
    df.to_excel(writer, index=False, sheet_name="All Metrics")

    workbook = writer.book
    # Define formats
    num_fmt = workbook.add_format({'num_format': '0.000'})
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2'})
    note_fmt = workbook.add_format({'italic': True, 'font_color': '#666666'})

    for metric, title in metrics:
        if metric not in df.columns: 
            continue
        lo, hi = metric+"_ci_lo", metric+"_ci_hi"

        # Build numeric pivot for values
        piv_val = df.pivot_table(index="trained_on", columns="tested_on", values=metric, aggfunc="mean")
        piv_val = piv_val.sort_index().sort_index(axis=1)

        # Build CI half-width table if available
        ci_hw = None
        if lo in df.columns and hi in df.columns:
            piv_lo = df.pivot_table(index="trained_on", columns="tested_on", values=lo, aggfunc="mean").sort_index().sort_index(axis=1)
            piv_hi = df.pivot_table(index="trained_on", columns="tested_on", values=hi, aggfunc="mean").sort_index().sort_index(axis=1)
            ci_hw = (piv_hi - piv_lo) / 2.0

        # Write numeric value table
        ws = workbook.add_worksheet(title[:31])
        writer.sheets[title[:31]] = ws

        # Write headers (with comma formatting)
        ws.write(0, 0, "Trained on \\ Tested on", header_fmt)
        col_labels = list(piv_val.columns)
        for j, c in enumerate(col_labels, start=1):
            lab = f"{int(c):,}" if isinstance(c,(int,float)) and not pd.isnull(c) else str(c)
            ws.write(0, j, lab, header_fmt)
        row_labels = list(piv_val.index)
        for i, r in enumerate(row_labels, start=1):
            lab = f"{int(r):,}" if isinstance(r,(int,float)) and not pd.isnull(r) else str(r)
            ws.write(i, 0, lab, header_fmt)

        # Write data cells with number format
        for i, r in enumerate(row_labels, start=1):
            for j, c in enumerate(col_labels, start=1):
                val = piv_val.loc[r, c]
                if pd.notnull(val):
                    ws.write_number(i, j, float(val), num_fmt)
                else:
                    ws.write_blank(i, j, None)

        # Apply green-yellow-red 3-color scale over the numeric data range
        if len(row_labels) > 0 and len(col_labels) > 0:
            ws.conditional_format(1, 1, len(row_labels), len(col_labels), {
                'type': '3_color_scale',
                'min_color': '#F8696B',  # red (low)
                'mid_color': '#FFEB84',  # yellow (mid)
                'max_color': '#63BE7B'   # green (high)
            })

        # Add a small note
        ws.write(len(row_labels)+2, 0, "Cells show metric value (3 d.p.). Green = higher, Yellow = mid, Red = lower.", note_fmt)

        # If CI half-width exists, write a paired sheet
        if ci_hw is not None:
            ci_title = (title + " (±CI)").strip()[:31]
            ws_ci = workbook.add_worksheet(ci_title)
            writer.sheets[ci_title] = ws_ci
            ws_ci.write(0, 0, "Trained on \\ Tested on", header_fmt)
            for j, c in enumerate(col_labels, start=1):
                lab = f"{int(c):,}" if isinstance(c,(int,float)) and not pd.isnull(c) else str(c)
                ws_ci.write(0, j, lab, header_fmt)
            for i, r in enumerate(row_labels, start=1):
                lab = f"{int(r):,}" if isinstance(r,(int,float)) and not pd.isnull(r) else str(r)
                ws_ci.write(i, 0, lab, header_fmt)
            for i, r in enumerate(row_labels, start=1):
                for j, c in enumerate(col_labels, start=1):
                    val = ci_hw.loc[r, c] if (r in ci_hw.index and c in ci_hw.columns) else np.nan
                    if pd.notnull(val):
                        ws_ci.write_number(i, j, float(val), num_fmt)
                    else:
                        ws_ci.write_blank(i, j, None)
            ws_ci.write(len(row_labels)+2, 0, "Cells show ± half-width of 95% CI (3 d.p.).", note_fmt)

out_path
