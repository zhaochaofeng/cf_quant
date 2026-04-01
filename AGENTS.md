# AGENTS.md - Coding Guidelines for cf_quant

## Project Overview

Quantitative finance research platform using Qlib framework. Python 3 codebase for factor modeling, backtesting, and portfolio analysis.

## Build/Lint/Test Commands

### Running Env
conda activate python311-tf210

### Code Execution
```bash
# Run Python scripts from repo root
python utils/preprocess.py
python data/process_data.py

# Run shell scripts
bash test.sh
```

### No Formal Test Framework
This project uses manual test files (`test.py`, `test2.py`) rather than pytest/unittest. Create test scripts that print validation results.

## Code Style Guidelines

### Imports
1. Standard library (sorted alphabetically)
2. Third-party packages: numpy, pandas, qlib
3. Local modules: `from utils import ...`, `from data.factor import ...`

```python
import os
from typing import Callable, Union

import numpy as np
import pandas as pd
from qlib.data import D

from utils import WLS, winsorize
```

### Formatting
- 4 spaces for indentation
- 2 blank lines between top-level functions/classes
- 1 blank line between methods
- Line length: ~100 characters
- Use single quotes for strings (consistent with existing code)

### Naming Conventions
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_CASE` for module-level constants
- Private methods: `_leading_underscore`

### Type Hints
Use Python 3 type hints for function signatures:

```python
def process_data(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    ...

def fetch_data(instruments: list[str], start_time: str) -> pd.DataFrame:
    ...
```

### Docstrings
Use Google-style docstrings with Chinese comments:

```python
def winsorize(data: pd.DataFrame, method: str = 'std') -> pd.DataFrame:
    """去极值处理
    
    Args:
        data: 输入数据
        method: 去极值方式，'std' 或 'quantile'
        
    Returns:
        pd.DataFrame: 去极值后的数据
    """
```

For complex functions, include Parameters/Returns sections in English for consistency with scientific libraries.

### Error Handling
- Use specific exceptions: `TypeError`, `ValueError`
- Include informative messages in Chinese:

```python
if not isinstance(data, pd.DataFrame):
    raise TypeError(f'不支持的数据类型: {type(data)}')
```

- Use try-except with logging for external API calls

### Data Processing Patterns
- Use pandas DataFrames with MultiIndex (instrument, datetime)
- Factor functions should accept df and return DataFrame with factor name as column
- Always sort by index: `df = df.sort_index()`
- Handle NaN values explicitly

### Configuration
- Store secrets in `config.yaml` (not committed)
- Use `utils.get_config()` to load configuration
- Never hardcode credentials

### Logging
- Use `LoggerFactory` from utils for structured logging
- Set appropriate log levels: DEBUG for development, INFO for production
- Include Chinese log messages for consistency

### Performance
- Use `multiprocessing_wrapper` for parallel computations
- Vectorize operations with pandas/numpy instead of loops
- Cache expensive computations using `qlib/cache.py`

### Git Practices
- Commit messages in English describing the change
- Don't commit large data files or cached results
- Keep `__pycache__` in .gitignore

## Dependencies

Key packages used:
- `qlib` - Quantitative finance framework
- `pandas`, `numpy` - Data processing
- `pymysql`, `sqlalchemy` - Database connectivity
- `redis` - Caching
- `tushare`, `jqdatasdk` - Data providers
- `baostock` - Alternative data source

Install via pip as needed (no requirements.txt present).

## Project Structure

```
/Users/chaofeng/code/cf_quant/
├── data/           # Data processing and factors
├── strategy/       # Trading strategies and backtesting
├── utils/          # Utility functions and connectors
├── qlib/           # Qlib extensions
├── test.py         # Manual tests
└── config.yaml     # Configuration (not committed)
```

Always import from `utils` using absolute imports from repository root.

---

## Operational Rules (Highest Priority)

### Git Commit/Push Workflow

**MANDATORY PROCESS - NO EXCEPTIONS:**

```
Development/Modification → Testing → Report Results → WAIT for User Confirmation → Execute Commit (after approval)
```

**STRICTLY PROHIBITED:**
- ❌ NEVER execute `git commit` without explicit user confirmation
- ❌ NEVER execute `git push` without explicit user confirmation
- ❌ NO exceptions (even for bug fixes, typos, or urgent changes)

**Confirmation Keywords:**
- User says "提交" (commit) → ✅ Can commit
- User says "确认" (confirm) → ✅ Can commit
- User says "推送" (push) → ✅ Can commit and push
- Silence/No response → ❌ Must wait
- "修改" or "修复" without explicit commit instruction → ❌ Must wait for confirmation

### Workflow Steps

**Step 1: Understand Requirements**
- Analyze user requirements
- Confirm technical approach
- Estimate timeline

**Step 2: Create Plan**
- Create TODO list
- Determine priority
- Estimate code volume

**Step 3: Execute Development**
- Write code following project standards
- Maintain code quality

**Step 4: Test and Verify**
- Run relevant tests
- Verify functionality
- Check output results

**Step 5: Report Results**
- Explain modifications
- Show test results
- Analyze potential impacts
- **WAIT for user confirmation**

**Step 6: Commit Code (AFTER approval)**
- Execute `git add`
- Execute `git commit` with descriptive message
- Execute `git push`
- Report completion

### Mode Independence

**Must follow commit workflow regardless of mode:**

- **Plan mode**: Create plan, report plan, wait for confirmation, then execute
- **Build mode**: Execute development, report results, wait for confirmation, then commit
- **Any mode**: Code commits require user confirmation

### Special Cases

**Bug Fixes:**
- Even obvious bugs require reporting and confirmation
- Cannot use "urgent fix" as excuse to skip confirmation

**Small Changes:**
- Even 1-line changes require reporting and confirmation
- No exceptions for small modifications

**Repeated Fixes:**
- Even if similar issues were fixed before, each fix requires confirmation
- Each modification is independent

### User Instruction Priority

**Highest Priority (Must Follow):**
1. User explicitly requires "report before commit"
2. User specifies commit workflow
3. User rejects/vetoes an operation

**Lower Priority:**
1. Project standards (AGENTS.md)
2. Code quality standards
3. Test coverage requirements

### Error Handling

**If commit workflow is violated:**
1. Acknowledge error immediately
2. Analyze violation cause
3. Explain what was committed
4. Propose remedy
5. Promise not to repeat

### Memory Anchors

**Must ask before commit:**
> "以上修改已完成并测试通过，是否确认提交到git仓库？"

**Must report after commit:**
> "已提交完成，提交哈希为 [hash]，推送至远程仓库"

---

**Last Updated**: 2026-03-15  
**Version**: v1.0  
**Must Follow**: YES (Highest Priority)
