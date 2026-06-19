import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 test runner fallback.
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]
ABFOLD_ROOT = ROOT / "abfold"
ABDIFF_ROOT = Path(os.environ.get("ABDIFF_REPO", ROOT.parent / "AbDiff"))


def parse_python(path):
    return ast.parse(path.read_text(encoding="utf-8", errors="replace"))


def module_path(module_name):
    if module_name == "abfold":
        return ABFOLD_ROOT / "__init__.py"
    if not module_name.startswith("abfold."):
        return None

    rel_parts = module_name.split(".")[1:]
    file_path = ABFOLD_ROOT.joinpath(*rel_parts).with_suffix(".py")
    package_path = ABFOLD_ROOT.joinpath(*rel_parts) / "__init__.py"

    if file_path.exists():
        return file_path
    if package_path.exists():
        return package_path
    return None


def hard_top_level_imports(path):
    imports = set()
    for node in parse_python(path).body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def guarded_imports(path):
    imports = set()
    for node in parse_python(path).body:
        if not isinstance(node, ast.Try):
            continue
        for body_node in node.body:
            if isinstance(body_node, ast.Import):
                imports.update(alias.name for alias in body_node.names)
            elif isinstance(body_node, ast.ImportFrom):
                imports.add(body_node.module or "")
    return imports


def exported_names(path):
    names = set()
    for node in parse_python(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def cleanup_build_artifacts():
    for name in ("build", "abfold_extended.egg-info"):
        path = ROOT / name
        if path.exists():
            shutil.rmtree(path)


class PyPIMVPTests(unittest.TestCase):
    def test_pyproject_names_distribution_separately_from_import_package(self):
        self.assertIsNotNone(tomllib, "tomllib is required to parse pyproject.toml")
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(data["project"]["name"], "abfold-extended")
        self.assertEqual(data["project"]["version"], "0.2.0")
        self.assertEqual(data["project"]["requires-python"], ">=3.8")
        self.assertEqual(data["tool"]["setuptools"]["packages"]["find"]["include"], ["abfold*"])

        package_data = data["tool"]["setuptools"]["package-data"]
        self.assertEqual(package_data["abfold.resources"], ["*.txt"])
        self.assertTrue((ABFOLD_ROOT / "resources" / "stereo_chemical_props.txt").exists())

    def test_import_package_exposes_version_without_heavy_imports(self):
        init_path = ABFOLD_ROOT / "__init__.py"
        self.assertEqual(hard_top_level_imports(init_path), set())
        self.assertIn("__version__", exported_names(init_path))

    def test_abdiff_referenced_abfold_modules_and_symbols_exist(self):
        if not ABDIFF_ROOT.exists():
            self.skipTest(f"AbDiff repo not found: {ABDIFF_ROOT}")

        files = [
            ABDIFF_ROOT / "abdiff" / "abfold_encoder" / "run_abfold_encoder.py",
            ABDIFF_ROOT / "abdiff" / "h3_mask" / "run_h3_mask.py",
            ABDIFF_ROOT / "abdiff" / "abdiff_sampling" / "nn" / "AF2" / "evoformer.py",
            ABDIFF_ROOT / "abdiff" / "abdiff_sampling" / "nn" / "AF2" / "triangular_attention.py",
            ABDIFF_ROOT / "abdiff" / "abdiff_sampling" / "nn" / "AF2" / "triangular_multiplicative_update.py",
            ABDIFF_ROOT / "abdiff" / "structure_decode" / "run_structure_decode.py",
        ]

        missing = []
        for file_path in files:
            self.assertTrue(file_path.exists(), f"Missing AbDiff file: {file_path}")
            for node in ast.walk(parse_python(file_path)):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith("abfold"):
                    continue

                base_path = module_path(node.module)
                if base_path is None:
                    missing.append(f"{file_path}: missing module {node.module}")
                    continue

                base_exports = exported_names(base_path)
                for alias in node.names:
                    candidate_module = f"{node.module}.{alias.name}"
                    if module_path(candidate_module) is not None:
                        continue
                    if alias.name not in base_exports:
                        missing.append(f"{file_path}: {alias.name} not found in {node.module}")

        self.assertEqual(missing, [])

    def test_optional_dependencies_are_not_hard_imports_for_mvp_paths(self):
        self.assertNotIn("abfold.diffusion.sample", hard_top_level_imports(ABFOLD_ROOT / "model.py"))
        self.assertNotIn("anarci", hard_top_level_imports(ABFOLD_ROOT / "data" / "data_process.py"))

        self.assertNotIn("deepspeed", hard_top_level_imports(ABFOLD_ROOT / "transition.py"))
        self.assertIn("deepspeed", guarded_imports(ABFOLD_ROOT / "transition.py"))

        self.assertNotIn("deepspeed", hard_top_level_imports(ABFOLD_ROOT / "utils" / "checkpointing.py"))
        self.assertIn("deepspeed", guarded_imports(ABFOLD_ROOT / "utils" / "checkpointing.py"))

    def test_train_ema_keeps_compat_device_helper_without_training_imports(self):
        train_imports = hard_top_level_imports(ABFOLD_ROOT / "train_ema.py")

        self.assertIn("abfold.utils.device", train_imports)
        self.assertIn("tensor_dict_to_device", exported_names(ABFOLD_ROOT / "train_ema.py"))

        forbidden = {
            "abfold.config",
            "abfold.data_module",
            "abfold.model",
            "abfold.utils.loss",
            "abfold.utils.unsupervised_loss",
            "torch.utils.tensorboard",
            "tqdm",
        }
        self.assertTrue(forbidden.isdisjoint(train_imports), sorted(forbidden & train_imports))

    @unittest.skipUnless(os.environ.get("ABFOLD_WHEEL_SMOKE") == "1", "set ABFOLD_WHEEL_SMOKE=1 to run wheel smoke test")
    def test_wheel_contains_abfold_package_and_resource_file(self):
        cleanup_build_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            wheel_dir = Path(tmp) / "wheel"
            install_dir = Path(tmp) / "install"
            wheel_dir.mkdir()
            install_dir.mkdir()

            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "--wheel-dir", str(wheel_dir)],
                    check=True,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                wheel = next(wheel_dir.glob("abfold_extended-0.2.0-*.whl"))
                with zipfile.ZipFile(wheel) as zf:
                    names = set(zf.namelist())
                    self.assertIn("abfold/__init__.py", names)
                    self.assertIn("abfold/training_config.py", names)
                    self.assertIn("abfold/resources/stereo_chemical_props.txt", names)
                    self.assertIn("abfold/utils/device.py", names)
                    self.assertFalse(any(name.endswith(".pyc") for name in names))

                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(install_dir), str(wheel)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                smoke = (
                    "import sys; "
                    f"sys.path.insert(0, r'{install_dir}'); "
                    "import abfold; "
                    "assert abfold.__version__ == '0.2.0'; "
                    "from abfold.training_config import config; "
                    "assert type(config).__name__ == 'TrainingConfig'"
                )
                subprocess.run([sys.executable, "-c", smoke], check=True)
            finally:
                cleanup_build_artifacts()


if __name__ == "__main__":
    unittest.main()
