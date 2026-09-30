# Code-size budget grants

`tests/test_code_size.py` fails when authored non-test source grows past its ceiling. The ceiling is the last `CEILING` line in that file plus the sum of the grants in this directory.

A change that grows the code on purpose adds one file here, named after its branch (for example `drivers-k3-serve.txt`), instead of editing the `CEILING` line:

```
+466
K3 TP16 serving path: per-rank KDA state, state budget, TP16 descriptor.
```

The first line is `+<lines>`; the lines after it justify the growth. Parallel pull requests then add different files and never conflict on the ceiling. A cleanup change may fold the grants into a new `CEILING` line and delete the files.
