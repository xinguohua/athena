import subprocess, sys, time

strategies = ["no_aug", "graphcl", "gca", "mimicry", "llm_guided"]
dataset = "theia"
scene = "theia311"
outfile = "bench_theia311_all_results.txt"

with open(outfile, "w") as f:
    f.write(f"===== {dataset}/{scene} 全部策略 =====\n")
    f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

for s in strategies:
    print(f"\n########## 策略: {s} ##########")
    cmd = [sys.executable, "-m", "process.benchmark_augmentation",
           "--strategy", s, "--dataset", dataset, "--scene", scene]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    print(output[-500:] if len(output) > 500 else output)
    with open(outfile, "a") as f:
        f.write(f"\n########## 策略: {s} ##########\n")
        f.write(output)
        f.write(f"\n########## {s} 完成 ##########\n")

print(f"\n全部完成，结果保存在 {outfile}")
