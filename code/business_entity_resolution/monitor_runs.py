#!/usr/bin/env python3
"""Single-window live monitor for the two BER background runs.

  * KAGGLE : upload -> push kernel -> poll -> download pipeline
  * LOCAL  : all-country streaming inference (France -> India -> US)

Both jobs run inside Claude's background shells; this script only *reads* their
output files, so quitting it (Ctrl-C) never stops the jobs.

Usage (any terminal with Python -- Git Bash, PowerShell, or cmd):
    python monitor_runs.py
    python monitor_runs.py <kaggle_output_file> <local_output_file>   # override
"""
import os, sys, time, re

TASKS = (r"C:\Users\ACER\AppData\Local\Temp\claude"
         r"\c--Users-ACER-OneDrive-Desktop-AmazonML"
         r"\9f705353-818f-4c61-a732-02312617a0e7\tasks")
KAGGLE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TASKS, "ba82ogir1.output")
LOCAL  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(TASKS, "b0kmv288j.output")

ORDER = ["France", "India", "US"]                       # processing order this run
CTOT  = {"France": 259452, "India": 809986, "US": 663106}
GRAND = sum(CTOT.values())                              # 1,732,544 test S1


def tail_text(path, nbytes=131072):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def read_full(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def lines_of(text):
    return [p for p in re.split(r"[\r\n]+", text) if p.strip()]


def done_flag(lines):
    for l in lines[-4:]:
        if "exited with code" in l:
            return l.strip()
    return ""


def kaggle_status(text):
    lines = lines_of(text)
    if not lines:
        return "(no output yet)", "", ""
    step = ""
    for l in lines:                        # scan full text for the latest step marker
        if l.startswith("== "):
            step = l
    return step, lines[-1], done_flag(lines)


def local_status(text):
    lines = lines_of(text)
    if not lines:
        return "(no output yet)", None, None, ""
    cur, n = None, None
    for l in reversed(lines):
        mm = re.search(r"\[(France|India|US)\]\s+([\d,]+)\s+S1", l)
        if mm:
            cur, n = mm.group(1), int(mm.group(2).replace(",", ""))
            break
    done = None
    if cur is not None:
        idx = ORDER.index(cur)
        done = sum(CTOT[ORDER[i]] for i in range(idx)) + min(n, CTOT[cur])
    return lines[-1], done, cur, done_flag(lines)


def bar(pct, width=40):
    fill = int(round(pct / 100.0 * width))
    return "#" * fill + "." * (width - fill)


def main():
    if os.name == "nt":
        os.system("")                      # enable ANSI escapes on Windows 10+
    while True:
        k_step, k_last, k_done = kaggle_status(read_full(KAGGLE))
        l_last, done, cur, l_done = local_status(tail_text(LOCAL))
        out = ["\x1b[2J\x1b[H"]
        out.append("=" * 78)
        out.append(" BER dual-run monitor    %s    (Ctrl-C quits; jobs keep running)"
                   % time.strftime("%H:%M:%S"))
        out.append("=" * 78)
        out.append("")
        out.append("[ KAGGLE  upload -> kernel ]   %s" % os.path.basename(KAGGLE))
        out.append("   step : %s" % (k_step or "(starting)"))
        out.append("   now  : %s" % k_last[:118])
        if k_done:
            out.append("   >>>> %s" % k_done)
        out.append("")
        out.append("[ LOCAL   all-country inference ]   %s" % os.path.basename(LOCAL))
        out.append("   now  : %s" % l_last[:118])
        if done is not None:
            pct = 100.0 * done / GRAND
            out.append("   test S1: %s/%s  [%s] %5.1f%%   (in: %s)"
                       % (format(done, ","), format(GRAND, ","), bar(pct), pct, cur))
        if l_done:
            out.append("   >>>> %s" % l_done)
        out.append("")
        out.append("-" * 78)
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        if k_done and l_done:
            print("\nBoth jobs have exited. Monitor stopping.")
            return
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(monitor stopped; background jobs are unaffected)")
