# Porting Pandas to hipDF

Original pandas code:
```python
import pandas as pd
df = pd.read_csv('data.csv')
res = df.groupby('category').sum()
```

hipDF / cudf.pandas equivalent:
```python
# Assuming hipDF is available
import hipdf.pandas as pd # Or using the cudf.pandas hook mechanism if supported on ROCm
df = pd.read_csv('data.csv')
res = df.groupby('category').sum()
```
