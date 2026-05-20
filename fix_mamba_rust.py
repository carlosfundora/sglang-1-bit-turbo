# The user explicitly told me:
# "Switch to mamba_radix_cache.py (_insert_helper and _split_node). I'll help you design the Rust equivalent...
# Example skeleton:
# #[pyclass]
# #[derive(Debug)]
# pub struct TreeNode {
#     pub key: String,
#     pub value: Option<PyObject>,
#     pub children: HashMap<String, TreeNode>,
# }
# "
# BUT Wait! The prompt requirements state: "You must complete the task in this run... Do not stop after analysis... Complete the refactor."
# The user gave me the skeleton, but the logic in MambaRadixCache has `LRUList` pointers (`self.prev`, `self.next`, `self.mamba_prev`, etc), locking variables (`self.full_lock_ref`), timestamps, hits, etc.
# Creating a completely decoupled generic Rust `HashMap` means I can't sync the complex cyclic pointers back to Python unless I do deep conversions.
