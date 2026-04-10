## 2024-04-10 - Import in hot loop
**Learning:** Found multiple instances where dynamic imports are placed inside the hot paths like `forward_hip` across various layers (`layernorm.py`, `activation.py`, `fp8_kernel.py`, `base.py`). Importing inside a hot loop is generally slower than importing outside or storing the imported functions lazily as global variables/class attributes.
**Action:** Lift these dynamic imports outside the hot loops or use global lazily initialized variables to avoid the `import` overhead on every single forward pass.
