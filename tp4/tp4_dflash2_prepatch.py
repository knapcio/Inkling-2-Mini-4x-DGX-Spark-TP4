"""TP pre-patch for incoai's DFlash2 SGLang compatibility patch (sglang_inkling_dflash2.py, revision 4f9088ea).

The pinned patch limits the drafter's candidate projection to the real vocabulary (200,058 of 201,024 padded ids)
and refuses a vocab-sharded target head ("validated for TP1"). Under TP the same limit is a per-shard clamp:
shard r owns ids [start_r, start_r + num_org_r), so it keeps the first clamp(valid - start_r, 0, num_org_r) of
them. With TP4 (4 x 50,256 rows) only the last shard is cut, to 49,290 rows; the union of the four shards' real ids
is exactly the TP1 set, so the gathered global top-k is the same candidate set.

Usage: python3 -I tp4_dflash2_prepatch.py <pinned patch> <output>. The pinned file is read, checked against its
sha256 and never written; the output is the same patch with the one guard replaced.
"""
import hashlib
import sys

PINNED_SHA256 = "14223ae2f3aa1d661da67cf0143ec33070d952a6e28a119e6cec64fcc0bebe50"
OLD = ("         '    if valid_vocab_size is not None:\\n'\n"
       "         '        if not tp:\\n'\n"
       "         '            raise ValueError(\"This Inkling benchmark integration is validated for TP1\")\\n'\n"
       "         '        num_org = min(num_org, int(valid_vocab_size))\\n'\n")
NEW = ("         '    if valid_vocab_size is not None:\\n'\n"
       "         '        if tp:\\n'\n"
       "         '            num_org = min(num_org, int(valid_vocab_size))\\n'\n"
       "         '        else:\\n'\n"
       "         '            num_org = max(0, min(num_org, int(valid_vocab_size)\\n'\n"
       "         '                                 - int(lm_head.shard_indices.org_vocab_start_index)))\\n'\n")


def prepatch(src: str) -> str:
    assert src.count(OLD) == 1, "TP1 guard not found exactly once: is this the pinned patch?"
    return src.replace(OLD, NEW)


if __name__ == "__main__":
    data = open(sys.argv[1], "rb").read()
    got = hashlib.sha256(data).hexdigest()
    if got != PINNED_SHA256:
        sys.exit(f"{sys.argv[1]}: sha256 {got} is not the pinned {PINNED_SHA256}")
    out = prepatch(data.decode())
    compile(out, sys.argv[2], "exec")
    open(sys.argv[2], "w").write(out)
    print(f"tp4_dflash2_prepatch: per-shard vocabulary clamp -> {sys.argv[2]}", file=sys.stderr)
