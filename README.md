# RL-gpu-smoketest

一套**极小规模**的 LLM 后训练 smoketest，唯一目的：练习在 VSC (KU Leuven, wICE 集群) 上申请 / 调度 GPU，并对比 V100 与 A100。

| Stage | 方法 | 模型 | 框架 |
|---|---|---|---|
| `check_gpu` | 打印 Slurm 分配到的 GPU 信息 + matmul 吞吐测试 | – | PyTorch |
| `sft_lora` | SFT + LoRA | Qwen3.5-0.8B | TRL `SFTTrainer` + PEFT |
| `grpo_lora` | GRPO（规则奖励：答案正确 + 格式）+ LoRA | Qwen3.5-0.8B | TRL `GRPOTrainer` + PEFT |
| `distill_onpolicy` | On-policy distillation（学生自采样，teacher 逐 token 打分，reverse KL）+ LoRA | Qwen3.5-2B → Qwen3.5-0.8B | TRL `DistillationTrainer` + PEFT |

数据：GSM8K 前 64 条。每个 stage 只跑 10–20 step，整个 job 在单卡上十几分钟内结束。

> 注：Qwen3.5 没有 0.7B，最小的是 `Qwen/Qwen3.5-0.8B`（实际 0.75B 参数）。可用环境变量 `STUDENT_MODEL` / `TEACHER_MODEL` 换模型。

精度自动选择（`smoketest/common.py: precision()`）：
- **A100 (sm_80)**：bf16 权重 + bf16 autocast
- **V100 (sm_70)**：硬件不支持 bf16 → fp32 权重 + fp16 autocast (AMP)
- 可用 `PRECISION=bf16|fp16|fp32` 强制覆盖

## 目录

```
env.sh                  # source 它：venv / HF 缓存 / 输出目录的路径
setup_env.sh            # 一次性：在 login 节点建 venv + 预下载模型和数据
requirements.txt
smoketest/
  common.py             # 精度、模型加载、LoRA、数据、summary.json
  check_gpu.py  sft_lora.py  grpo_lora.py  distill_onpolicy.py
  run_all.sh            # 依次跑所有 stage（batch job 和交互式都用它）
  summarize.py          # 汇总 summary.json 成表格（对比 V100 vs A100）
slurm/
  smoketest.slurm       # batch job 脚本（#SBATCH 默认值 = A100 debug）
  submit.sh             # slurm/submit.sh v100|a100|a100-debug
  interactive.sh        # 交互式拿一张卡
```

存储位置（VSC 惯例）：`$VSC_HOME` 只有 3 GB，所以 venv 和 HF 缓存放 `$VSC_DATA`，训练输出放 `$VSC_SCRATCH/rl-smoketest/<jobid>/`。

---

# 教程：在 VSC 上预约 / 使用 GPU

## 0. 一次性准备（login 节点）

```bash
ssh vsc38501@login.hpc.kuleuven.be          # 或通过 Open OnDemand 的 shell
cd ~/workspace/RL-gpu-smoketest              # 或 git clone 本仓库
bash setup_env.sh                            # 建 venv + 下载 Qwen3.5-0.8B/2B + GSM8K（约 6 GB，放在 $VSC_DATA）
```

> **login 节点只用来编辑、安装、下载、提交作业，不要在上面训练。** login 节点限制了线程数（直接跑 torch 会报 `Thread creation failed`）。

## 1. 基本概念

- **集群**：KU Leuven Tier-2 有 `wice`（GPU 都在这里）和 `genius`。所有 Slurm 命令都要带 `-M wice`（或 `--clusters=wice`），否则可能看不到你的作业。
- **Credits（计费账户）**：GPU 作业必须 `--account=<项目>`。你的账户是 `intro_vsc38501`（200 万 credits）。
  ```bash
  sam-balance                                   # 查余额
  sam-quote sbatch -p gpu_v100 -c 4 -t 01:00:00 slurm/smoketest.slurm   # 提交前估价（按申请的最长时间算）
  sam-statement                                 # 账单明细
  ```
  实际按**真实运行时间**扣费，所以 `--time` 不要虚报太多——但写太短作业会被杀。
- **Partition（队列）**：不同 GPU 在不同 partition。`sinfo -M wice -s` 看全部。和本项目相关的：

| Partition | GPU | 每节点 | 每卡建议 CPU | 最长时间 | 1h 报价* |
|---|---|---|---|---|---|
| `gpu_v100` | Tesla V100-SXM2 32GB (sm_70) | 8 卡 / 36 核 | 4 | 3 天 | ~4 680 |
| `gpu_v100_long` | 同上 | | 4 | 7 天 | |
| `gpu_a100` | A100-SXM4 80GB (sm_80) | 4 卡 / 72 核 | 18 | 3 天 | ~11 220 |
| `gpu_a100_debug` | A100-PCIe 80GB | 1 卡 / 64 核 | ≤16 | **1 小时** | ~10 738 |
| `interactive` | A100 的 MIG 切片 (1g.10gb) | | ≤8 | 16 小时 | |
| `gpu_h100` | H100 80GB | 4 卡 | | 3 天 | |

\* 用 `sam-quote` 在 2026-10-30 查得，单卡、按申请时间上限计。

- **资源请求**：GPU 用 `--gpus-per-node=N`（或 `--gres=gpu:N`），CPU 用 `--cpus-per-task`，内存默认 = CPU 数 × 每核内存（V100 节点 21 GB/核，A100 节点 7 GB/核），一般不用单独写 `--mem`。**按卡的比例要 CPU**（V100: 4 核/卡，A100: 18 核/卡），要多了会挤占同节点其他卡、也多扣钱。

## 2. 方式一：批处理作业 `sbatch`（推荐）

```bash
cd ~/workspace/RL-gpu-smoketest
slurm/submit.sh v100          # 交到 V100
slurm/submit.sh a100          # 交到 A100
slurm/submit.sh a100-debug    # 交到 A100 debug 队列（≤1h，通常最快开始）
```

`submit.sh` 实际执行的就是：

```bash
sbatch --account=intro_vsc38501 --partition=gpu_v100 --cpus-per-task=4 --time=01:00:00 slurm/smoketest.slurm
```

`slurm/smoketest.slurm` 里的 `#SBATCH` 行是默认值；命令行参数优先级更高。额外参数直接追加：

```bash
slurm/submit.sh v100 --time=00:20:00                          # 改时间
slurm/submit.sh a100 --export=ALL,STAGES="grpo_lora"           # 只跑 GRPO
slurm/submit.sh a100 --begin=now+2hours                        # 2 小时后才开始排队（"定时"）
slurm/submit.sh a100 --begin=2026-11-02T09:00                  # 指定时间之后开始
slurm/submit.sh v100 --mail-type=END,FAIL --mail-user=you@example.com   # 结束时发邮件
sbatch --test-only -M wice -A intro_vsc38501 -p gpu_a100 -c 18 slurm/smoketest.slurm  # 只估计开始时间，不真提交
```

## 3. 方式二：交互式拿一张卡（调试用）

```bash
slurm/interactive.sh a100-debug      # 等价于 srun -M wice -A intro_vsc38501 -p gpu_a100_debug -c 16 --gpus-per-node=1 -t 00:59:00 --pty bash -l
# 拿到节点后（提示符变成 k28i22 之类）：
nvidia-smi
bash smoketest/run_all.sh                         # 跑全部
STAGES="sft_lora" bash smoketest/run_all.sh       # 只跑一个
exit                                              # 一定要 exit，否则一直计费到 --time 用完
```

另一个选择：浏览器打开 **Open OnDemand**（https://ondemand.hpc.kuleuven.be），用 "Interactive Shell" / "Jupyter Lab" 应用，表单里选 partition、GPU 数、时长即可，原理一样是 Slurm 作业。

## 4. 监控与管理

```bash
squeue -M wice -u $USER                   # 我的作业：PENDING / RUNNING
squeue -M wice -u $USER --start           # PENDING 作业的预计开始时间
scontrol -M wice show job <jobid>         # 详细信息（Reason、分到的节点、GPU）
tail -f logs/rl-smoketest-<jobid>.out     # 实时日志
ssh <节点名>                               # 作业运行时可登录到计算节点，然后 nvidia-smi / gpuwatch / htop
scancel -M wice <jobid>                   # 取消作业
sacct -M wice -j <jobid> --format=JobID,Partition,Elapsed,State,AllocTRES%60   # 结束后的记录
seff <jobid>                              # 结束后的 CPU / 内存利用率
```

常见 PENDING 原因（`squeue` 的 NODELIST(REASON) 列）：
- `Resources`：卡都被占着，排队等。A100 很抢手，V100 通常空闲更多；`gpu_a100_debug` 只有 1 张卡。
- `Priority`：前面有优先级更高的作业。
- `QOSMaxGPUPerUserLimit` / `AssocGrpBillingMinutes`：超出每用户上限 / credits 不够。

每个作业的产物：`$VSC_SCRATCH/rl-smoketest/<jobid>/`，里面有每个 stage 的 `summary.json`、LoRA adapter、以及每 5 秒采样一次的 `gpu_usage.csv`。

## 5. 对比 V100 vs A100

```bash
source env.sh
python -m smoketest.summarize $VSC_SCRATCH/rl-smoketest/*
```

实测结果（2026-10-30，job 62474749 = V100，62474750 = A100 debug；单卡，GSM8K 64 条）：

| Stage | GPU | 精度 | s/step | 峰值显存 GB | loss 首→末 | 末步 reward | 跳过步数* |
|---|---|---|---|---|---|---|---|
| sft_lora (20 步) | V100-SXM2 32GB | fp16 | 1.07 | 3.88 | 1.092 → 0.435 | – | 7 |
| sft_lora (20 步) | A100 80GB PCIe | bf16 | 0.50 | 2.26 | 1.092 → 0.415 | – | 0 |
| grpo_lora (10 步) | V100-SXM2 32GB | fp16 | 10.8 | 4.37 | – | 1.075 | 5 |
| grpo_lora (10 步) | A100 80GB PCIe | bf16 | 10.3 | 2.69 | – | 0.70 | 0 |
| distill_onpolicy (10 步) | V100-SXM2 32GB | fp16 | 6.21 | 11.11 | 0.227 → 0.213 | – | 0 |
| distill_onpolicy (10 步) | A100 80GB PCIe | bf16 | 5.67 | 5.75 | 0.219 → 0.214 | – | 0 |

整个 job：V100 4 分 35 秒，A100 4 分 04 秒（含模型加载）。

\* fp16 AMP 的 GradScaler 在梯度溢出时会跳过该步并降低 loss scale（日志里 `grad_norm=nan`），前几步常见、属正常；bf16 没有这个问题。

怎么读这张表：
- SFT 是纯前向/反向，A100 约快 2×（matmul 实测：V100 fp16 95 TFLOPS，A100 bf16 252 TFLOPS）。
- GRPO / 蒸馏的大部分时间花在逐 token 的 `generate` 上（还是 Qwen3.5 线性注意力的纯 PyTorch fallback），受 kernel 启动延迟限制，所以两种卡差不多。要发挥 A100 需要 vLLM 和 `flash-linear-attention`。
- V100 上 fp32 权重 + fp16 AMP 比 A100 bf16 多用约一倍显存，但 0.8B/2B 都远没到上限。
- GRPO 的 reward 和 loss 在 10 步里噪声很大，这里只检验"有学习信号"（grad_norm > 0），不代表效果。

## 6. 关于"预约"（reservation）

Slurm 里"预约"有三种含义：
1. **排队**（上面的 `sbatch` / `srun`）：这就是日常用法，资源空出来就自动开始。
2. **延后开始**：`--begin=<时间>`，作业到点之后才参与排队，不保证到点立即有卡。
3. **真正的资源预留**（`scontrol create reservation`）：只有管理员能建。需要在某个时间段保证拿到 GPU（比如课程、deadline），要发邮件给 VSC helpdesk（hpcinfo@kuleuven.be）申请，批下来后用 `--reservation=<名字>` 提交。

## 7. V100 / A100 需要注意的差异

- V100 不支持 bf16：代码自动改用 fp32 权重 + fp16 AMP，速度和显存都比 A100 bf16 差。
- 新版 PyTorch 的 CUDA 12.8+ wheel 逐步去掉了老架构，所以本项目装 **cu126** 版 torch——已确认 `torch.cuda.get_arch_list()` 同时包含 `sm_70` 和 `sm_80`，同一个 venv 两种卡都能用。
- 为了两种卡用同一套代码，rollout 用 transformers 自带的 `generate`，没有用 vLLM（慢，但 smoketest 足够）。
- Qwen3.5 的线性注意力层（Gated DeltaNet）在没装 `flash-linear-attention` / `causal-conv1d` 时会退回纯 PyTorch 实现（日志里会有 `falling back` 警告），能跑但慢，这对 smoketest 无所谓。
