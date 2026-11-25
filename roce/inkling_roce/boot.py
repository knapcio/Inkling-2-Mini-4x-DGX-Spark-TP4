# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-import patcher: apply a patch to each target module right after it executes.

Unchanged from our MiniMax-M3.1 recipe's ``mm31_roce.boot`` except that importing this
module does nothing by itself (no ``.pth`` loader, no environment switch here): the SGLang
hook module (``tp4/inkling_sgl_roce.py``) calls ``install_hooks`` when INKLING_TP4_ROCE=1.
The patcher works in whichever process imports a target (the HTTP server, the scheduler
and detokenizer subprocesses), because ``sitecustomize`` runs in every interpreter.

A patch that fails raises: the operator asked for RoCEnante, so a half-patched engine
must not come up.  ``verify_targets`` in the in-image preflight catches drift earlier.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import threading


class PostImportPatcher(importlib.abc.MetaPathFinder):
    """Run ``patch(module)`` right after each target module executes (once per name)."""

    def __init__(self, targets):
        self._targets = dict(targets)
        self._busy = threading.local()

    def pending(self):
        return sorted(self._targets)

    def find_spec(self, name, path=None, target=None):
        if name not in self._targets:
            return None
        busy = getattr(self._busy, "names", None)
        if busy is None:
            busy = self._busy.names = set()
        if name in busy:
            return None  # our own find_spec call below
        busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            busy.discard(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        patch = self._targets.pop(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                patch(module)
            except Exception as exc:
                raise RuntimeError(f"INKLING_TP4_ROCE: patching {name} failed: {exc!r}") from exc

        spec.loader.exec_module = exec_and_patch
        return spec

    def patch_already_imported(self):
        """Apply patches for targets imported before the hook existed."""
        for name in list(self._targets):
            module = sys.modules.get(name)
            if module is not None:
                self._targets.pop(name)(module)


def install_hooks(targets):
    """Install the post-import patcher; returns the finder."""
    finder = PostImportPatcher(targets)
    sys.meta_path.insert(0, finder)
    finder.patch_already_imported()
    return finder
