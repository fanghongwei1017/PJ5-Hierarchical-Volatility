# 图表与排版源码

`reports/` 保存三份提交文件及其 LaTeX 源码；共用的 `figs/` 包含图形定义、绘图 CSV 和表格。主稿、完整股票数据附录和八股曲线附录均可直接从这些固定文件编译，无须重新拟合模型。

## 编译正式稿的副本

需要带 `ctex`、`pgfplots` 的 XeLaTeX 环境。下列 PowerShell 命令从仓库根目录执行，将输出放在仓库旁的一个新目录，保留正式 PDF 不变：

```powershell
$projectRoot = (Resolve-Path '.').Path
$typesetRun = Join-Path (Split-Path $projectRoot -Parent) 'pj5-typeset-check'
if (Test-Path -LiteralPath $typesetRun) { throw '目标目录已存在，请换一个新目录名。' }
Copy-Item -LiteralPath (Join-Path $projectRoot 'reports') -Destination $typesetRun -Recurse
Push-Location -LiteralPath $typesetRun
try {
    foreach ($document in @('report', 'data_supplement', 'eight_stock_supplement')) {
        1..2 | ForEach-Object {
            xelatex -interaction=nonstopmode -halt-on-error "$document.tex"
            if ($LASTEXITCODE -ne 0) { throw "编译失败：$document" }
        }
    }
} finally { Pop-Location }
```

提交的三份 PDF 分别为 18、25、4 页。数据附录首页保留了此前的题名，其统计内容对应当前主稿的 8∶1∶1 划分。

## 从实验结果重新计算八股图表

八股图表使用各股的逐点评分、每日状态和已保存参数；不重新训练。历史生成脚本的输出路径固定在旧版目录中，因此要在独立副本运行。使用实验环境中的 Python 3.13.9、NumPy 2.3.5、SciPy 1.16.3、pandas 2.3.3，并将 `$py` 设置为该 Python 的绝对路径。

```powershell
$py = (Get-Command python).Source  # 须指向上述实验环境
$projectRoot = (Resolve-Path '.').Path
$figureRun = Join-Path (Split-Path $projectRoot -Parent) 'pj5-figures-check'
if (Test-Path -LiteralPath $figureRun) { throw '目标目录已存在，请换一个新目录名。' }
Copy-Item -LiteralPath (Join-Path $projectRoot 'research') -Destination $figureRun -Recurse
New-Item -ItemType Directory -Path (Join-Path $figureRun '20261115_八股对照版/figs') | Out-Null
Push-Location -LiteralPath $figureRun
try {
    & $py -I -B -X utf8 '工作文件/八股图表_20261115/eight_figures.py'
    if ($LASTEXITCODE -ne 0) { throw '八股图表计算未通过，请检查输出。' }
} finally { Pop-Location }
```

生成的 CSV 和四张组合图源码位于副本的 `20261115_八股对照版/figs/`；数值检查及来源清单位于副本的 `依据/八股图表_20261115/`。正式提交的 `reports/figs/` 不受影响。

不要把 `--restyle` 当作重复核验入口：它是一次性排版修订步骤，要求恰好有四个 TeX 文件发生变化，已完成该修订的版本不满足这一条件。

其余保留的历史绘图脚本用于追溯图表生成过程，其中可能引用旧报告目录、原始行情或缓存；不能将它们视为在当前目录直接重跑的入口。正式图文的完整排版入口是本页的 XeLaTeX 命令，实验复算入口见仓库 README 及两个实验包的复算说明。

## 主稿中的湍流类比与两项实证图

第 1、2、6 页分别画出研究顺序、三维湍流的尺度传递和股票模型的计算流程。第 9 页的八股窗口相关图读取逐股点估计和分块区间。第 10 页左图把训练期的滚动方差与单层、八层的背景分布放在同一坐标系中，右图比较给定方差的三条高斯曲线与积分后的八层收益密度。第 14 页的两幅图分别显示中国平安的日背景与收益分布，第 15 页给出八股日背景的跨期比较。绘图数据的生成方法见[复算说明](REPRODUCING.md)。

## 检查文件是否完整

在仓库根目录运行：

```powershell
python -X utf8 tools/verify_submission.py
```

该工具只读取文件，核对提交清单及两套原实验清单的文件大小和 SHA-256，不会重新拟合或覆盖结果，也不代替科学计算的独立核验。仅此文件检查工具使用 Python 标准库，需要 Python 3.11 或更新版本。
