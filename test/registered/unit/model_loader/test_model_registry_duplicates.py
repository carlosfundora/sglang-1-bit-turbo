import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import torch.nn as nn

from sglang.srt.models import registry


class TestModelRegistryDuplicates(unittest.TestCase):
    def tearDown(self):
        registry.import_model_classes.cache_clear()

    def test_duplicate_model_names_keep_first(self):
        package_name = f"fakepkg_dup_{uuid.uuid4().hex[:8]}"

        with TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / package_name
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "mod_one.py").write_text(
                "import torch.nn as nn\n"
                "class JinaBertModel(nn.Module):\n"
                "    pass\n"
                "EntryClass = JinaBertModel\n",
                encoding="utf-8",
            )
            (package_dir / "mod_two.py").write_text(
                "import torch.nn as nn\n"
                "class JinaBertModel(nn.Module):\n"
                "    pass\n"
                "EntryClass = JinaBertModel\n",
                encoding="utf-8",
            )

            sys.path.insert(0, temp_dir)
            try:
                registry.import_model_classes.cache_clear()
                imported = registry.import_model_classes(package_name)
                self.assertIn("JinaBertModel", imported)
                self.assertTrue(issubclass(imported["JinaBertModel"], nn.Module))
            finally:
                sys.path.pop(0)
                for name in list(sys.modules):
                    if name == package_name or name.startswith(f"{package_name}."):
                        sys.modules.pop(name, None)
