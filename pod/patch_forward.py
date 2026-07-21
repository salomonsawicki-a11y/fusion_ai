#!/usr/bin/env python3
"""
Idempotently patch the upstream tokamind bug: mmt/eval/forward.py uses
TYPE_CHECKING-guarded names (TorchDecoder) in runtime function signatures, which
raises NameError on import. Inserting `from __future__ import annotations` right
after the module docstring makes all annotations lazy and fixes the import.

Patches the file of the *installed* mmt package (editable installs point back into
the checkout), then hard-verifies the import.

The file is located via find_spec WITHOUT importing mmt.eval — importing it is
exactly what crashes before the patch is applied.
"""

import ast
import importlib
import importlib.util
import os
import sys

spec = importlib.util.find_spec("mmt")
if spec is None or spec.origin is None:
    sys.exit("cannot locate installed mmt package")
target = os.path.join(os.path.dirname(spec.origin), "eval", "forward.py")
src = open(target).read()

if "from __future__ import annotations" in src:
    print(f"already patched: {target}")
else:
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    # Insert after the module docstring if present, else at the top.
    insert_at = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        insert_at = tree.body[0].end_lineno
    lines.insert(insert_at, "\nfrom __future__ import annotations\n")
    open(target, "w").write("".join(lines))
    print(f"patched: {target}")

importlib.invalidate_caches()
import mmt.eval.forward  # noqa: F401,E402

print("import mmt.eval.forward: OK")
sys.exit(0)
