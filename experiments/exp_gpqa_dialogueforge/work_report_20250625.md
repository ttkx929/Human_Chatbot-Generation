# 实验开发文档：GPQA Diamond 多轮对话数据合成（DialogueForge 扩展）

> 日期：2025-06-25（更新：数据质量改进 v2）  
> 仓库：[ttkx929/Human_Chatbot-Generation](https://github.com/ttkx929/Human_Chatbot-Generation)（基于 [nerchio/Human_Chatbot-Generation](https://github.com/nerchio/Human_Chatbot-Generation) fork）  
> 状态：**管线 v2 已完成开发与 5 条试跑验证；全量 198 条需用新策略重跑**

---

## 1. 实验目的

| 问题 | 说明 |
|------|------|
| 当前问题 | 需要高质量的多轮对话 SFT 数据；单轮 QA 或 Baize 式一次生成难以覆盖「追问—纠错—归纳」的辅导过程 |
| 实验目标 | 以 **GPQA Diamond**（198 道高难科学 MCQ）为种子，用 **DialogueForge 双智能体**流程合成多轮「学生 ↔ 科学导师」对话，导出可微调格式（ShareGPT / Baize） |
| 方案来源 | DialogueForge（[arXiv:2507.15752](https://arxiv.org/abs/2507.15752)）+ Baize 数据合成思路 + 自研 GPQA 接入与工程修复 |
| 成功标准 | ① 管线可端到端复现；② 对话以导师（bot）结尾、科学内容可抽检；③ 全量 198 条种子 + 多轮扩写完成；④ 微调后科学 benchmark 有提升（**待验证**） |

```text
本阶段成功标准（当前）：
- 数据合成脚本可复现：已通过
- 全量 198 题种子生成（v1 LLM seed）：已完成（需用 v2 策略重生成）
- 首轮泄题率可控（<15%）：v2 试跑 5 条 0/5，待全量验证
- 模型训练与 GPQA 评测：未开始
```

---

## 2. 论文或方案分析

### 2.1 DialogueForge 解决什么问题

真实多轮人机对话数据稀缺。DialogueForge 用两个 LLM 交替扮演 **Inquirer（人类）** 与 **Responder（助手）**，从种子对话扩写成长对话，再可用于微调。

### 2.2 本实验怎么解决（相对原仓库的扩展）

| 环节 | 原 DialogueForge | 本实验扩展（v1） | v2 质量改进 |
|------|------------------|------------------|-------------|
| 种子来源 | OASST / Arena 真实对话摘要 | **GPQA Diamond** 科学 MCQ → 2 轮种子 | 多样化 human seed + **template tutor seed（默认）** |
| Prompt | 通用闲聊 | `prompts_gpqa.py` 科学辅导场景 | **分阶段 spoiler 规则**、accuracy 规则、学生口吻约束 |
| 微调目标 | 默认偏 Inquirer（人类） | **Responder（科学导师）** | 同上；首轮不泄题，中后段才确认选项 |
| API | 多模型同时初始化 | **DeepSeek 按需加载** | DeepSeek `temperature` 降至 **0.3** |
| 数据质量 | — | **对话必须以 bot 结尾** | **`quality_checks.py` + `--filter-early-spoiler`** |
| 教学模式 | — | — | 随机 `pedagogy_mode`：scaffold / walkthrough / misconception |

### 2.3 与 Baize 的区别

| | Baize | 本方案 |
|--|-------|--------|
| 种子 | Quora 等单问 | GPQA 科学题 + 标准答案元数据 |
| 生成 | 单模型一次生成整段 | LangGraph 双 agent 交替，追问更可控 |
| 领域 | 通用聊天 | 生物 / 物理 / 化学研究生难度 |

---

## 3. 论文复现实验

> 本节为原 DialogueForge 论文复现；**未做**

| 项目 | 内容 |
|------|------|
| 复现目标 | TBD（如 OASST 6-turn / 12-turn 生成与 UniEval） |
| 模型 | TBD |
| 数据 | TBD |
| 结论 | TBD |

---

## 4. 自己数据训练说明

### 4.1 数据集说明

| 数据集 | 数量（当前） | 目标数量 | 用途 |
|--------|-------------:|---------:|------|
| GPQA Diamond 种子（v1 LLM seed） | 198 | — | 旧策略，首轮泄题率高，**建议废弃** |
| GPQA Diamond 种子（v2 template seed） | 5（试跑） | 198 | 每题 2 轮种子 + `pedagogy_mode` 元数据 |
| 多轮对话（v1 生成结果） | 20 | — | 旧数据，首轮泄题 ~65%，仅作对比 |
| 多轮对话（v2 生成结果） | 5（试跑） | 198 | LangGraph 扩写，max_turns=12 |
| SFT 导出（ShareGPT，v2） | 5 | 198 | `gpqa_diamond_sft.json`，经 `--filter-early-spoiler` |

**数据流（v2 推荐）：**

```text
GPQA Diamond (HF: Idavidrein/gpqa, config=gpqa_diamond)
    → prepare_gpqa_seed.py  [--seed-model none]   # 默认 template seed，不向 LLM 喂正确答案字母
    → data/gpqa_diamond_seed.jsonl                # metadata 含 pedagogy_mode
    → main.py  [--data gpqa_diamond]              # responder 按 assistant_turns 动态切换 prompt
    → results/gpqa_diamond_DeepSeek_DeepSeek_12.jsonl
    → sft_reformat_gpqa.py  [--format sharegpt --filter-early-spoiler]
    → data/gpqa_diamond_sft.json                  # metadata.quality.early_spoiler 标签
```

### 4.2 数据质量对比（v1 vs v2，试跑 5 条同题 gpqa_diamond_0~4）

| 指标 | v1（旧策略，20 条统计） | v2（新策略，5 条试跑） |
|------|------------------------|------------------------|
| 严格首轮泄题（`early_spoiler`） | **13/20（65%）** | **0/5（0%）** |
| 以 bot 结尾 | 20/20（修复后） | 5/5 |
| 平均总轮数 | 11.3 | 12.0 |
| 平均 bot 回复长度 | ~1673 字符 | ~1301 字符 |
| 学生 AI 腔短语（生成轮） | 0.23/轮 | 0.04/轮 |
| seed 首条泄题 | LLM seed 约 12/20 抽检命中 | template seed 0/5 |
| 导出过滤后保留 | 无过滤 | 5/5 通过（若 v1 加同样过滤约剩 ~35%） |

**v2 仍存在的已知问题：**

- template seed 开场 5/5 完全相同，规模化后需增加 seed 变体
- 试跑 5 条 `pedagogy_mode` 均为 scaffold，walkthrough / misconception 待全量验证
- 「软泄题」仍存在：部分对话在中途用数值/「唯一可行选项」收窄答案，但未写字母
- 个别题目 bot 长回复内部逻辑偶有不一致（如自旋期望値题的纠错段落）

### 4.3 数据配比配置

> 当前为**单数据源**实验，未做混合配比；后续若与 Baize / 通用 SFT 混合再填。

```yaml
# TBD — 示例占位
datasets:
  gpqa_diamond_sft:
    path: Generation/data/gpqa_diamond_sft.json
    ratio: 1.0   # 当前仅科学辅导数据
    filter:
      early_spoiler: true   # 导出时启用 --filter-early-spoiler
```

---

## 5. 训练配置

> **未启动训练**，以下为目标方案占位。

| 参数 | 值 |
|------|-----|
| Base Model | TBD（如 Qwen2.5-7B / Llama-3-8B） |
| Training Type | SFT（LoRA） |
| 训练框架 | TBD（baize-chatbot `finetune.py` 或 LLaMA-Factory） |
| 数据格式 | ShareGPT `messages` 或 Baize `[|Human|]/[|AI|]` |
| Max Length | TBD（科学对话较长，建议 ≥2048） |
| Learning Rate | TBD |
| Epoch | TBD |

**SFT 训练建议（基于 v2 数据特性）：**

- 优先使用 `--filter-early-spoiler` 导出后的样本
- 可考虑 `--format assistant_turns`，丢弃前 1–2 轮 assistant 标签，进一步降低泄题污染
- 训练前人工抽检 10–20 条，重点看有机/对称性等易错机制题

---

## 6. 训练过程记录

> 未开始训练，无曲线。

```text
curves/
├── train_loss.png          # TBD
├── learning_rate.png       # TBD
├── gpu_memory.png          # TBD
└── throughput.png          # TBD
```

**数据生成过程记录：**

| 阶段 | 命令 / 改动 | 状态 |
|------|-------------|------|
| v1 依赖修复 | `requirements.txt` Windows 可安装版 | 完成 |
| v1 全量种子 | `prepare_gpqa_seed.py --seed-model DeepSeek`（198 题） | 完成（**建议用 v2 重跑**） |
| v1 试跑 + 导出 | 20 条对话 + ShareGPT 导出 | 完成（质量基线） |
| v1 结尾修复 | `conversation_utils.py` + `fix_dialogue_endings.py` | 完成 |
| **v2 prompt 改进** | `prompts_gpqa.py`：spoiler / accuracy / 学生口吻 / pedagogy_mode | **完成** |
| **v2 seed 改进** | `prepare_gpqa_seed.py`：默认 `--seed-model none`，多样化 human seed | **完成** |
| **v2 扩写改进** | `main.py`：按 `assistant_turns` 动态切换 responder prompt | **完成** |
| **v2 质量门禁** | `quality_checks.py` + `sft_reformat_gpqa.py --filter-early-spoiler` | **完成** |
| **v2 试跑** | `--limit 5` 全流程，`Early spoiler rate: 0/5` | **完成** |
| v2 全量对话 | `main.py`（无 limit，198 题） | **待开展** |
| 文档 / 推送 | README、本报告、git commit | 进行中 |

---

## 7. 评测结果

> 未微调，无评测表。注意：**不可用 GPQA Diamond 作评测集**（与训练同源，会污染）。

| Benchmark | Base | Fine-tune | Δ |
|-----------|-----:|----------:|--:|
| GPQA Diamond | — | — | （不作为评测集） |
| 其他科学推理集 | TBD | TBD | TBD |
| MMLU（科学子集） | TBD | TBD | TBD |

**数据质量评测（非 benchmark，供内部参考）：**

| 指标 | v1 | v2（5 条试跑） |
|------|----:|---------------:|
| `early_spoiler` 比例 | 65% | 0% |
| 综合 SFT 可用性（主观） | ~6.8/10 | ~8.0/10 |

---

## 8. 实验结论

```text
1. 是否有效：
   v1：数据合成管线有效，但首轮泄题严重（~65%），不适合直接训「辅导型助教」。
   v2：试跑 5 条显示质量改进有效 —— 首轮泄题归零，学生人设更自然，确认答案时机后移。

2. 为什么：
   - v1 根因：LLM seed 向模型喂入 Known correct option；responder prompt 鼓励首轮讲对错选项。
   - v2 改动：template seed + 分阶段 spoiler 规则 + 导出过滤 + 低温生成。
   - 剩余风险：template 重复、软泄题、部分机制题叙事偏移、全量规模未验证。

3. 下一步：
   ① 用 v2 命令重跑全量 198 题（seed + main + sft_reformat）
   ② 全量统计 early_spoiler 率；若 >10% 收紧第 2–3 轮 responder 规则
   ③ 给 build_bot_seed_template() 增加 3–4 个变体，消除 seed 同质化
   ④ 扩展 quality_checks：检测「软泄题」（唯一选项 + 数值锁定）
   ⑤ 选定 base 模型，启动 LoRA SFT；在非 GPQA 科学 benchmark 上评测
   ⑥ 人工抽检 10–20 条，重点核对有机合成 / 对称性 / 遗传互作等易错题
```

---

## 附录 A：关键文件清单

| 文件 | 说明 |
|------|------|
| `Generation/data/prepare_gpqa_seed.py` | GPQA → 种子 jsonl；v2 默认 template seed、pedagogy_mode |
| `Generation/prompts_gpqa.py` | 科学场景 prompt；v2 分阶段 spoiler / accuracy 规则 |
| `Generation/main.py` | 支持 `gpqa_diamond`、bot 结尾、动态 assistant_turns prompt |
| `Generation/data/quality_checks.py` | **v2 新增**：首轮泄题检测 |
| `Generation/data/conversation_utils.py` | 对话结尾工具；closing 轮 spoiler 约束 |
| `Generation/data/sft_reformat_gpqa.py` | 导出 ShareGPT / Baize；`--filter-early-spoiler` |
| `Generation/data/fix_dialogue_endings.py` | 修复已有 jsonl（v1 遗留，v2 图内已处理） |
| `Generation/setting.yaml` | DeepSeek temperature 0.3 |
| `Generation/run_gpqa_pipeline.ps1` | v2 流程示例 |
| `Generation/models.py` | 模型懒加载、DeepSeek 默认 |
| `.gitignore` | 忽略 api.yaml、生成数据 |

---

## 附录 B：复现命令

### B.1 试跑（5 条，推荐先验证）

```powershell
cd E:\LLM\Human_Chatbot-Generation\Generation
.\.venv\Scripts\Activate.ps1

python data/prepare_gpqa_seed.py --source huggingface --seed-model none --limit 5

python main.py --data gpqa_diamond --inquirer_model DeepSeek --responder_model DeepSeek --max_turns 12 --limit 5

python data/sft_reformat_gpqa.py `
  --input results/gpqa_diamond_DeepSeek_DeepSeek_12.jsonl `
  --output data/gpqa_diamond_sft.json `
  --format sharegpt `
  --filter-early-spoiler
```

### B.2 全量（198 题）

```powershell
python data/prepare_gpqa_seed.py --source huggingface --seed-model none --output data/gpqa_diamond_seed.jsonl

python main.py --data gpqa_diamond --inquirer_model DeepSeek --responder_model DeepSeek --max_turns 12

python data/sft_reformat_gpqa.py `
  --input results/gpqa_diamond_DeepSeek_DeepSeek_12.jsonl `
  --output data/gpqa_diamond_sft.json `
  --format sharegpt `
  --filter-early-spoiler
```

### B.3 可选：仍用 LLM 写 seed（不推荐，泄题风险更高）

```powershell
python data/prepare_gpqa_seed.py --source huggingface --seed-model DeepSeek
```

> 注意：`Generation/api.yaml` 含 API 密钥，已在 `.gitignore` 中，**切勿提交到 Git**。
