"""Static rule checks for solution/ (standard library only, no torch).   python3 tools/lint_rules.py solution/

Fails (exit 1) and prints file:line: reason for:
  - imports of ready-made kernel libraries (list read from BANNED_MODULES in harness/common.py)
  - @triton.autotune
  - CUDAGraph / make_graphed_callables / torch.compile
  - os.environ / os.getenv (behaviour must not depend on the environment)
  - an @triton.jit kernel never launched as kernel[grid](...) anywhere in solution/ (idle kernel)
  - kernels named example_* outside solution/swiglu.py (copies of the shipped example)
A line with `# lint: ok` is skipped. check_rules.py on the T4 does the deeper checks (renamed copies).
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = "# lint: ok"


def banned_modules():
    tree = ast.parse((ROOT / "harness" / "common.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "BANNED_MODULES" for t in node.targets):
            return set(ast.literal_eval(node.value))
    sys.exit("lint_rules: BANNED_MODULES not found in harness/common.py")


def is_jit(dec, jit_names):
    node = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(node, ast.Attribute):
        return node.attr == "jit" and getattr(node.value, "id", "") in {"triton"}
    return isinstance(node, ast.Name) and node.id in jit_names


def attr_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "solution")
    target = target if target.is_absolute() else Path.cwd() / target
    files = sorted(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
    banned = banned_modules()
    problems, kernels, launched = [], {}, set()

    def add(path, line, text, lines):
        if IGNORE not in lines[line - 1]:
            problems.append(f"{path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}:{line}: {text}")

    for path in files:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{path}:{exc.lineno}: does not parse ({exc.msg})")
            continue
        jit_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "triton":
                jit_names |= {a.asname or a.name for a in node.names if a.name == "jit"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in banned:
                        add(path, node.lineno, f"imports banned library {a.name}", lines)
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in banned:
                    add(path, node.lineno, f"imports banned library {node.module}", lines)
                if root == "os" and any(a.name in ("environ", "getenv") for a in node.names):
                    add(path, node.lineno, "os.environ / os.getenv is not allowed in solution/", lines)
            elif isinstance(node, ast.Attribute):
                chain = attr_chain(node)
                if chain in ("os.environ", "os.getenv"):
                    add(path, node.lineno, f"{chain} is not allowed in solution/ (no behaviour that depends on the environment)", lines)
                if node.attr == "autotune":
                    add(path, node.lineno, "triton autotune is not allowed (the compile cache is empty in every run)", lines)
                if node.attr in ("CUDAGraph", "make_graphed_callables") or chain.endswith("cuda.graphs"):
                    add(path, node.lineno, f"CUDA graphs are not allowed ({node.attr})", lines)
                if chain == "torch.compile":
                    add(path, node.lineno, "torch.compile in solution/: its kernels do not count as ours; remove or mark `# lint: ok`", lines)
            elif isinstance(node, ast.Name) and node.id in ("CUDAGraph", "make_graphed_callables", "autotune"):
                add(path, node.lineno, f"{node.id} is not allowed", lines)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(is_jit(d, jit_names) for d in node.decorator_list):
                    kernels[node.name] = (path, node.lineno, lines)
                    if node.name.startswith("example_") and path.name != "swiglu.py":
                        add(path, node.lineno, f"kernel {node.name} copies the shipped example (example_* names belong to swiglu.py only)", lines)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                f = node.func.value
                launched.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    for name, (path, line, lines) in sorted(kernels.items()):
        if name not in launched:
            add(path, line, f"@triton.jit kernel {name} is never launched as {name}[grid](...) in solution/ (idle kernels are forbidden)", lines)
    for p in problems:
        print(p)
    print(f"lint_rules: {len(files)} files, {len(kernels)} kernels, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
