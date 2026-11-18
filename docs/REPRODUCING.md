# 复算说明

从仓库根目录运行。以下 `$py` 应指向 Python 3.13.9 且安装了冻结依赖的解释器；例如激活 `environment.yml` 所建环境后：

```powershell
$py = (Get-Command python).Source
& $py -X utf8 -c "import platform,numpy,scipy,pandas; print(platform.python_version(),numpy.__version__,scipy.__version__,pandas.__version__)"
if ($LASTEXITCODE -ne 0) { throw '实验环境检查失败。' }
& $py -X utf8 tools/verify_submission.py
if ($LASTEXITCODE -ne 0) { throw '提交文件核验失败。' }
```

## 条件波动实验

快速核验保存结果：

```powershell
Push-Location research
try {
    & $py -I -B -X utf8 '工作文件/条件波动拓展_20261114/package_extension.py' --verify
    if ($LASTEXITCODE -ne 0) { throw '条件实验核验失败。' }
} finally { Pop-Location }
```

这会检查原始交付清单、因果性、密度预检、半衰期选择、八股的每日状态和参数来源，不重新训练。输出写入新建的 `.package_verification_*` 目录，该目录不提交到 Git。

重做完整独立科学核验，可将 `--verify` 换成 `--validate-full`。详细范围及新拟合命令见 [条件实验原始说明](../research/依据/条件波动拓展_20261114/README_复算.md)。执行逐点高精度核验必须使用该说明中的 `validate_csv_schema.py`，不要调用冻结脚本的 `--stage all`；其旧校验入口有 CSV 列类型推断问题。

## 静态拟合、扩层与辅助统计

原始核验器会重新写入校验摘要，因此先把 `research/` 复制到仓库外的一个新目录，再执行：

```powershell
$dest = Join-Path (Split-Path (Get-Location).Path -Parent) 'pj5-static-check'
if (Test-Path -LiteralPath $dest) { throw '目标已存在，请另选新目录。' }
Copy-Item -LiteralPath research -Destination $dest -Recurse
Push-Location $dest
try {
    & $py -B -X utf8 '工作文件/811重划_20261114/fit_validate.py'
    if ($LASTEXITCODE -ne 0) { throw '静态拟合核验失败。' }
    & $py -B -X utf8 '工作文件/811重划_20261114/aux_validate.py'
    if ($LASTEXITCODE -ne 0) { throw '辅助统计核验失败。' }
} finally { Pop-Location }
```

核验涉及 12 组拟合的训练尺度、窗口、分期评分与扩层停止条件，以及训练日重抽样、时钟相关和日内统计。全新拟合可用原说明中的 `--fresh-copy` 创建仅含输入和代码的副本；详见 [静态实验原始说明](../research/依据/811重划实验_20261114/README_复算.md)。

## 不重叠窗口与方差混合图

在仓库根目录执行下列命令，输出写入仓库旁的复算目录。

```powershell
$newResults = Join-Path (Split-Path (Get-Location).Path -Parent) 'pj5-intraday-recheck'
& $py -B -X utf8 research/experiments/session_persistence.py --output (Join-Path $newResults 'persistence')
if ($LASTEXITCODE -ne 0) { throw '相邻窗口实验未完成。' }
& $py -B -X utf8 research/experiments/variance_mixture.py --output (Join-Path $newResults 'mixture')
if ($LASTEXITCODE -ne 0) { throw '方差混合曲线未完成。' }
```

第一个实验读取八股的历史与 2026 年五分钟收益，分别在上午、下午比较两个相邻的 40 分钟窗口。先按固定钟点选齐窗口，再按交易日作 2,000 次循环分块重抽。24 行结果（八股、三个时期）、输入指纹及绘图数据见 [相邻窗口实验](../research/experiments/results/session_persistence/)。

第二个计算读取中国平安的训练收益和既有拟合参数，重建 18,528 个局部方差估计，用单层解析密度核对数值卷积。输出的三个 CSV 与 [主稿绘图数据](../reports/figs/) 中同名文件对应。两条命令均可单独运行。

## 检查范围

`tools/verify_submission.py` 检查文件是否与提交快照一致；科学核验脚本检查输入、参数和结果的数值关系。两者都不等于重新训练。初版打包的检查结果保存在 `docs/validation/package_checks.json`，原实验的独立核验随结果保存在 `research/依据/` 内。

全新拟合的耗时、文件元数据可能变化；应比较参数、评分和协议的数值约束，不能要求所有新结果文件与旧文件哈希相同。
