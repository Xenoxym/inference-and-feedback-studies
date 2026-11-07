# 本机推理实验说明

本实验检验固定目标模型与后端下，历史复用和起草预算是否换来实际时间收益。它不是 Adaptix、DISCO 或 BanditSpec 的复现，也未训练新算法。

## 运行

在研究根目录用 Python 3.12 执行：

```powershell
python experiments/inference/rerun_benchmark.py --repeats 5 --tokens 64 --output results/inference_rerun
python scripts/run_sampling_audit.py --output results/sampling_audit_rerun.json
```

rerun_benchmark.py仅为原测量脚本指定新结果目录，避免覆盖交付日志；benchmark.py本身保持测量时的SHA256。run_sampling_audit.py 在临时目录执行原采样脚本，并将结果写入指定新文件。原 sampling_audit.py 保留测量时版本。

已执行环境为 torch 2.12.0+cpu、transformers 5.11.0、float32、SDPA attention、CPU 4计算线程/1互操作线程、batch=1。所有策略使用同一 SmolLM2-135M 和 DynamicCache，逐请求重置KV；拒绝后的缓存尾部用crop裁掉。模型版本、权重SHA256、软件版本、输入和测时口径保存在results/inference/metadata.json。依赖不要求使用GPU，原作者Rust检索器也不参与本实验。

模型目录是sources/SmolLM2-135M。复跑前可用download_model.py按已固定SHA下载，无需登录或提供私人数据。脚本只读该模型目录，不执行远程模型代码。

## 预算与历史定义

普通解码 ar 不维护草稿表。fixed4/fixed8 最多起草4/8个token，随后同一次target forward验证anchor与草稿。adaptive从4开始，预算全满且全接受时加2，否则减1，限制1至8；是简单的接受反馈启发式，不是成本优化器，也不会主动关闭speculation。fixed4/8未用开发集优化，不能称为最优固定基线。

三元组表由token频次构成，选择频次最高后继，并按token ID打破平局。它每次只接受已提交token的更新，与Adaptix的MCTS、最近窗口反复加权和多分支验证不同。

- prompt_only：每请求只有输入prompt的统计，生成后不更新。
- request：输入prompt加本请求已提交输出，每个新三元组只更新一次。
- cross：固定的3条外部先行请求及其输出作为初始历史，再加本请求历史。每个测试请求均重新复制同一历史；不会积累其他测试请求答案。它不是完整在线跨请求服务模拟。

六个公开合成prompt覆盖叙述、科学解释、代码、问答、格式化重复、显式重复。它们是诊断输入，不是随机抽取的代表性benchmark。测试参考输出仅由verifier和一致性检查使用，不提供给提议器。先用独立prompt进行运行时warmup，且每个新请求重置提议表，避免warmup答案泄漏到测试。

## 正确性与计时

每个测试prompt的普通解码输出先与同后端model.generate的贪心结果核对。所有方法、历史范围与重复运行再逐token对照该参考。每轮保存draft、accepted、committed、discarded、budget和分段时间。EOS或最大长度触发停止。

warm_request_s包含表初始化、prefill与decode的墙钟时间；排除tokenization、模型下载/加载、结果序列化。decode_s包含Python循环控制。CPU调用同步，使用perf_counter。first_request.json单列加载后首次请求，metadata单列导入/加载时间；不是冷进程分布的重复估计。prior_records.json保存外部历史建设过程，不将这笔一次性成本藏入跨请求速度结论。

每个重复将6提示×3历史×4方法的72个任务顺序打乱，共5重复360请求。汇总先按“同重复、同历史、同方法”的6提示总耗时求和，再计算AR/方法的比值；报告五个比值中位数及范围。它不是6个任务speedup的算术平均。五次是时钟重复，不是增加了独立任务数量。桌面系统仍有其他正常应用，因此不把小差别作统计显著性声称。

分段时间中的draft_s包含选择公共anchor和草稿搜索；commit_s包含匹配比较、写入已提交序列、表更新及控制器；rollback_s单列KV裁剪。failed_round_verify_s属于“发生拒绝的完整轮”的目标验证时间，其中也包含有用计算，不能与verify_s相加，也不能称为可精确剥离的纯浪费时间。

## 可复核边界

候选末端正好填满输出上限时，此实现可能计算一个不会再用到的最后logit，而AR会跳过最后单token forward；这会保守地压低推测方法性能，不破坏输出一致性。rounds包含不执行forward的最终shortcut；模型调用次数应按非shortcut轮数计算，另加prefill。

原脚本EOS统计有一个未触发的边界：anchor已经是EOS时仍可能先构造draft，或draft中间到EOS时把后续截断记作failed。最终分析必须确认所测记录是否出现EOS，且不得据未覆盖分支声称通用停止逻辑已经充分验证。本次若未出现EOS，应明确报告未覆盖；不要静默修改历史日志。输出一致性检查仍会发现任何测试输出错误。

synthetic_replay.json先于真实模型运行产生。它用完整隐藏参考轨迹在不同轮边界重新提议并验证，成本1+0.15k仅为明示假设的cost units，不是毫秒或实测加速。sampling_audit.py则是独立的采样正确性验证，调用固定作者版本真实verifier，按其评估入口补全失败后的重采样；不测LLM速度。


公开版复核归档推理日志可运行 `python scripts/verify_inference_results.py --results results/inference`。
