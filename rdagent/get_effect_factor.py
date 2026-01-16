'''
    因子迭代后，获取有效的因子信息
'''

import pickle
from pathlib import Path

def extract_valid_factors_only(log_path: str | Path):
    """从日志目录只提取有效因子"""
    log_path = Path(log_path)

    # 加载 session
    session_path = log_path / "__session__"
    # 按照每轮每个步骤排序。
    # 如files[0]为: /data/RD-Agent/log/2025-12-14_11-34-24-080469/__session__/0/0_direct_exp_gen
    files = sorted(session_path.glob("*/*_*"),
                   key=lambda f: (int(f.parent.name), int(f.name.split("_")[0])))

    with open(files[-1], "rb") as f:
        # <class 'rdagent.app.qlib_rd_loop.factor.FactorRDLoop'>
        loop = pickle.load(f)

    valid_factors = []

    for exp, exp_feedback in loop.trace.hist:
        # 1. 跳过整体失败的实验
        if not (exp_feedback and exp_feedback.decision):
            continue

        # 2. 获取每个子因子的反馈
        sub_feedbacks = exp.prop_dev_feedback  # CoSTEERMultiFeedback
        if sub_feedbacks is None:
            continue

        # 3. 遍历每个因子，检查其单独的反馈
        for i, (task, ws, fb) in enumerate(zip(
                exp.sub_tasks,
                exp.sub_workspace_list,
                sub_feedbacks  # 可迭代，返回 CoSTEERSingleFeedback
        )):
            # 关键：检查单个因子的 final_decision
            if fb is None or not fb.final_decision:
                continue  # 跳过无效因子

            if ws is None or "factor.py" not in ws.file_dict:
                continue  # 跳过没有代码的因子

            valid_factors.append({
                "name": task.factor_name,
                "description": task.factor_description,
                "formulation": task.factor_formulation,
                "variables": task.variables,
                "code": ws.file_dict["factor.py"],
                "hypothesis": exp.hypothesis.concise_justification if exp.hypothesis else None,
                # 可选：包含反馈信息
                "execution_log": fb.execution,
                "return_checking": fb.return_checking,
            })

    return valid_factors


if __name__ == "__main__":
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else "log"

    # 找到最新的日志
    log_dir = Path(log_path)
    if log_dir.is_dir() and not (log_dir / "__session__").exists():
        log_dir = sorted(log_dir.iterdir())[-1]

    factors = extract_valid_factors_only(log_dir)

    print(f"共找到 {len(factors)} 个 **有效** 因子:\n")
    for i, f in enumerate(factors, 1):
        print(f"{'=' * 50}")
        print(f"因子 {i}: {f['name']}")
        print(f"描述: {f['description']}")
        print(f"公式: {f['formulation']}")
        print(f"代码预览:\n{f['code'][:300]}...")

    '''
        python get_effect_factor.py /data/RD-Agent/log/2025-12-14_11-34-24-080469
    '''

