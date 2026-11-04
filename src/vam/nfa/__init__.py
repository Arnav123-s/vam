"""The nested fractal architecture (spec N, ``spec/VAM_Nested_Fractal_Architecture.md``).

The owner's seven layers, each made a working part of the machine (DESIGN section 10):

* ``bounds``: layer 1 and the size limits of layers 3 and 5. The exact Bekenstein and
  Bekenstein-Hawking formulas, and the machine's own register, cache line and caches read from
  the hardware. Every size the architecture uses (a context label, a density matrix, an MPO bond)
  is capped by these numbers.
* ``page``: layer 2. Page's average entropy of a random state, the edge of a random state's
  spectrum (Marchenko-Pastur), and the rank cutoff they give: a direction is kept only if it
  stands above what a random (Page) state would show.
* ``flow``: layer 1. Every event of the machine written as one differential equation
  ``dx/dt = f(x, t)`` and solved by an ODE integrator, plus the trajectory's Page-sized basis.

Later layers (readout, blankets, governor, collapse) are added next to these.
"""
