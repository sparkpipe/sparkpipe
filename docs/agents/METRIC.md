
## The metric: maximize Solutions / (code size SQUARED)

Every proposed change must state which SOLUTION it buys:

- **DRY/structural wins** (remove duplicated code): always right - deleting a
  line is a solution at zero cost. Do these first.
- **Performance levels** are distinct solutions with distinct values, in
  this order: accurate-but-slow (1) < 80% of SOTA (2) < 90% of SOTA (3)
  < matches SOTA (4) < exceeds SOTA (5). Moving a driver up one level IS
  a solution and justifies more code; code that does not buy a level is
  not a solution.

Rule: writing code to buy a performance level is good; writing code for
its own sake is not. When you propose an improvement, name the level it
buys (or state it is a DRY win) and estimate the code-size delta (+/-).

SQUARED, not times-two: code size costs quadratically. +100 lines must buy 10,000 units of solution value. Deletions are superlinear wins.
