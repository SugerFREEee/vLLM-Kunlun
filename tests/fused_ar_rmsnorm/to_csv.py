import os
import re
import csv

LOG_DIR = "."
OUT_FILE = "summary.csv"

# 正则匹配三个指标
patterns = {
    "throughput": re.compile(r"Output token throughput \(tok/s\):\s*([\d.]+)"),
    "ttft": re.compile(r"Mean TTFT \(ms\):\s*([\d.]+)"),
    "tpot": re.compile(r"Mean TPOT \(ms\):\s*([\d.]+)")
}

def parse_filename(filename):
    # 解析 benchmark_in1024_out1024_b16_n16.log
    m = re.fullmatch(r"benchmark_in(\d+)_out(\d+)_b(\d+)_n(\d+)\.log", filename)
    if m:
        return int(m.group(3)), int(m.group(1)), int(m.group(2))
    return None, None, None

results = []

for fname in os.listdir(LOG_DIR):
    if not fname.endswith(".log"):
        continue

    path = os.path.join(LOG_DIR, fname)

    conc, input_len, output_len = parse_filename(fname)

    throughput = None
    ttft = None
    tpot = None

    with open(path, "r") as f:
        content = f.read()

        # 提取指标
        if patterns["throughput"].search(content):
            throughput = patterns["throughput"].search(content).group(1)

        if patterns["ttft"].search(content):
            ttft = patterns["ttft"].search(content).group(1)

        if patterns["tpot"].search(content):
            tpot = patterns["tpot"].search(content).group(1)

    results.append({
        "concurrency": conc,
        "input_len": input_len,
        "output_len": output_len,
        "throughput": throughput,
        "ttft": ttft,
        "tpot": tpot
    })

# 排序（按并发 + 输入长度）
results.sort(key=lambda x: (x["concurrency"], x["input_len"]))

# 写入 CSV
with open(OUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "concurrency", "input_len", "output_len",
        "throughput", "ttft", "tpot"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ 已生成 {OUT_FILE}")