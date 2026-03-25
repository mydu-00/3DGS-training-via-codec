## Plan: Legacy Server + Agent Workflow

在不升级课题组服务器系统（Ubuntu 18.04, 无 sudo）的前提下，采用“本地新版 VS Code + Copilot Agent（主控）+ 远程 SSH 执行（算力）”双端工作流，先保证今天可用，再尝试恢复 Remote-SSH 完整体验（通过官方 old-linux sysroot workaround）。

**Direction**
1. 服务器信息：
   - 服务器地址：10.210.3.133
   - 用户名：qianjunhong
   - 密码：Qwer123456
2. 要求：
   - 不要动里面的文件，重新从github下一个高斯的工程（inria）
   - 环境配好了，直接用gs那个环境,micromamba activate gs
   - 记得改一下下载的工程命名，别原来工程重了，覆盖了就玩完了
3. 任务内容
    3.1. 先跑一版3dgs原版的base，记录SSIM,PSNR,LPIPS这三个指标。
    3.2. 在训练阶段，每一次迭代，backward 3DGS参数更新后，存回global memory之前，将3dgs的颜色球谐函数sh部分参数量化成int8，然后再反量化回float32，这两组都循环30k次，记录SSIM,PSNR,LPIPS这三个指标，看一下掉了多少。
    3.4. 训练设置，python train.py -s xx/tandt/truck --eval

**Steps**
1. Phase 1 - 先验证硬约束与最小可用链路
1.1 在远程服务器执行并记录环境：`ldd --version`、`strings $(g++ -print-file-name=libstdc++.so.6) | grep GLIBCXX | tail`、`uname -r`，确认是否低于 `glibc 2.28 / GLIBCXX 3.4.25`。  
1.2 在本地保留最新版 VS Code + GitHub Copilot（Agent 可用），不再走“降级 VS Code”路线。  
1.3 建立“单终端远程执行入口”：本地 PowerShell/Windows Terminal 登录远程 + `tmux` 会话固定，后续所有训练、调试、日志都在该会话内。

1. Phase 2 - 搭建你要的“本地描述 -> agent改代码 -> 远程跑并看结果”
2.1 代码同步基线（推荐 Git-first）：本地与远程都基于同一仓库；功能分支开发；本地 agent 负责改代码并提交；远程 `git pull` 后执行训练。  
2.2 若网络/权限导致 Git 不便，备用为 rsync/scp 增量同步脚本（仅同步源码与配置，不同步大数据和 checkpoint）。  
2.3 远程标准运行入口：统一 `run_train.sh`、`run_eval.sh`、`run_debug.sh`（或等价命令），便于 agent 以后稳定生成可复用命令。  
2.4 远程日志产物规范：`logs/`、`artifacts/`、`checkpoints/` 分离；训练命令统一 `tee` 输出，便于本地 agent 读取失败栈和性能数据后继续修改。  
2.5 针对 3DGS/PyTorch CUDA：优先让 agent 走“读报错 -> 最小修复 -> 远程重跑单测/短跑 -> 再全量训练”的闭环，而不是直接长跑。

1. Phase 3 - 尝试恢复 VS Code Remote-SSH（可选增强，依赖 Phase 1）
3.1 依据官方 old-linux workaround 准备用户态 sysroot（glibc>=2.28 + libstdc++>=3.4.25）与 patchelf>=0.18（都放 `~/tools`，无需 root）。  
3.2 在远程用户环境设置 3 个变量：`VSCODE_SERVER_CUSTOM_GLIBC_LINKER`、`VSCODE_SERVER_CUSTOM_GLIBC_PATH`、`VSCODE_SERVER_PATCHELF_PATH`。  
3.3 本地 Remote-SSH 先清理旧 server，再重连验证（这是“技术性绕过”，非官方完全支持路径，升级后可能需重复适配）。

1. Phase 4 - 3DGS 项目专项稳定化
4.1 固化环境导出：`pip freeze`（或 conda env export）与 CUDA/PyTorch版本记录，避免“今天能跑明天不能跑”。  
4.2 Submodule 与依赖一次性检查清单：`git submodule status`、编译扩展步骤、必要环境变量。  
4.3 建立“短基准任务”（几十秒到几分钟）用于 agent 快速验证改动正确性，长训练只在短基准通过后触发。  
4.4 性能/显存回归门槛：固定一个场景记录 step time、显存峰值、关键指标，防止 agent 修改导致隐性退化。

1. 并行关系与依赖
5.1 `Phase 1` 与 `Phase 2` 可并行起步，但 `Phase 2` 的稳定性依赖 `Phase 1` 的环境确认。  
5.2 `Phase 3` 完全可选，且不阻塞你日常开发。  
5.3 `Phase 4` 在 `Phase 2` 具备基本闭环后立即开展。

**Relevant files**
- `~/.ssh/config`（本地）— 维护远程别名、端口、跳板配置，减少每次连接复杂度。
- `repo_root/run_train.sh`（远程仓库）— 统一训练入口参数与日志重定向。
- `repo_root/run_eval.sh`（远程仓库）— 统一验证入口，支持短基准。
- `repo_root/scripts/sync_to_remote.ps1`（本地可选新建）— Git 不便时的增量同步。
- `repo_root/docs/remote_workflow.md`（仓库文档，可选新建）— 记录“agent本地+远程执行”操作约定。

**Verification**
1. 连接验证：本地 SSH 登录后 10 秒内进入固定 `tmux` 会话，无人工重复操作。
2. 闭环验证：本地用 agent 完成一次最小代码修改 -> 推送/同步到远程 -> 远程短基准跑通 -> 回收日志并定位结果。
3. 故障验证：故意引入一个可控报错（如参数拼写错），确认 agent 能根据远程日志自动给出修复并再次通过。
4. 可选 Remote-SSH 验证：若配置 sysroot 路线，确认 Remote-SSH 可进入工作区且不再报 glibc/libstdc++ 缺失。

**Decisions**
- 已确认：你有用户目录权限（`~/`），无 sudo。
- 已确认：你的核心目标不是“必须 Remote-SSH”，而是“保留 Copilot Agent 复杂任务能力并驱动远程训练调试闭环”。
- 包含范围：工作流重构、兼容性 workaround 路线、3DGS 调试提速策略。
- 不包含范围：服务器系统升级、管理员级别安装、改变课题组基础设施策略。

**Further Considerations**
1. 推荐优先级：Option A 先落地 Phase 1+2（当天可用）；Option B 再尝试 Phase 3（恢复远程 IDE 体验）。
2. 工作流定位：这不是 GitHub CI/CD 自动流水线，而是“本地 Agent 主导开发 + 远程机器执行训练”的人机协作循环。

**30分钟操作清单（先能用）**
1. 本地保持最新版 VS Code + Copilot Agent；远程只用 SSH 终端，不强求 Remote-SSH 插件连接成功。
2. 本地和远程都使用同一个 Git 仓库（同一分支策略）。
3. 远程进入项目后先完成一次环境自检：Python、CUDA、PyTorch、submodule、关键依赖。
4. 远程准备一个短跑命令（几十秒到几分钟），用于每次改动后的快速验证。
5. 本地向 Agent 提需求并改代码；本地提交后推送。
6. 远程拉取最新提交，执行短跑命令并保存日志。
7. 若失败，把错误日志贴给 Agent，让它最小化修复；重复第5-6步直到短跑通过。
8. 短跑通过后再启动长训练，并在固定目录保存输出指标。

**与CI/CD的关系**
1. CI/CD 偏“自动化发布与持续集成”，通常由 GitHub Actions 等在云端触发。
2. 你当前最需要的是“研究型开发循环”：人主导目标，Agent 辅助改代码，远程 GPU 主导执行验证。
3. 之后可增量加一个轻量 CI：只跑 lint/单测/格式检查，不跑完整 GPU 训练。