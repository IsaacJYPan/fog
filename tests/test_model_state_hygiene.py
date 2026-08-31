"""Model forwards are pure with respect to batch-specific diagnostic state."""

from __future__ import annotations

import ast
from pathlib import Path


def test_forward_methods_do_not_assign_attributes_on_self() -> None:
    models = Path(__file__).parents[1] / "fog" / "models"
    violations: list[str] = []
    for path in models.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "forward":
                continue
            for child in ast.walk(node):
                targets = []
                if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = (
                        child.targets
                        if isinstance(child, ast.Assign)
                        else [child.target]
                    )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        violations.append(f"{path.name}:{child.lineno}:{target.attr}")
    assert violations == []
