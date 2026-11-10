"""Run every experiment listed in a job file, in parallel over the available GPUs.

Usage: python experiments/run_jobs.py WORKDIR [JOBFILE] [--workers N] [--filter PREFIX]

WORKDIR must contain cache/ (from prep_data.py). Results are written to WORKDIR/res2/NAME.json and
logs to WORKDIR/logs/NAME.log. Jobs whose result file already exists are skipped, so the script can
be restarted after an interruption.
"""
import argparse, os, queue, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("workdir")
ap.add_argument("jobfile", nargs="?", default=os.path.join(HERE, "jobs.txt"))
ap.add_argument("--workers", type=int, default=2, help="parallel runs (e.g. 2 per GPU)")
ap.add_argument("--filter", default="", help="only run jobs whose name starts with this prefix")
a = ap.parse_args()

W = os.path.abspath(a.workdir)
ngpu = len(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.strip().splitlines()) or 1
jobs = [l.strip().split("|", 1) for l in open(a.jobfile) if l.strip() and not l.startswith("#")]
jobs = [j for j in jobs if j[0].startswith(a.filter)]
q = queue.Queue()
for j in jobs:
    q.put(j)
print(len(jobs), "jobs,", a.workers, "workers,", ngpu, "GPU(s)", flush=True)


def worker(k):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % ngpu))
    while True:
        try:
            name, args = q.get_nowait()
        except queue.Empty:
            return
        out = os.path.join(W, "res2", name + ".json")
        if os.path.exists(out):
            continue
        log = os.path.join(W, "logs", name + ".log")
        os.makedirs(os.path.dirname(out), exist_ok=True); os.makedirs(os.path.dirname(log), exist_ok=True)
        t = time.time()
        cmd = [sys.executable, os.path.join(HERE, "..", "otofl.py"), "--cache", os.path.join(W, "cache"), "--out", out] + args.split()
        with open(log, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        print(time.strftime("%H:%M"), name, f"{time.time() - t:.0f}s", "ok" if os.path.exists(out) else "FAILED",
              "left", q.qsize(), flush=True)


ths = [threading.Thread(target=worker, args=(k,)) for k in range(a.workers)]
for t in ths:
    t.start()
for t in ths:
    t.join()
