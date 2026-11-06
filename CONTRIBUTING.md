# Contributing

Start with the development commands in the README. Keep changes focused, add regression
tests for behavioral fixes, and run Ruff, Black, pytest, and the CPU demo before opening
a pull request. Changes to tokenization or metric definitions must explain their effect
on score comparability.

Use synthetic or explicitly licensed sample documents. Do not commit private documents,
credentials, model weights, checkpoints, or local tracking directories. Report experimental
results with the input hash, split protocol, runtime environment, and evaluation settings.
Label unmeasured expectations clearly and keep them out of benchmark tables.

A useful pull request explains the concrete problem, resulting behavior, validation, and
any limits on what was verified. GPU changes should identify the GPU and CUDA environment
used, or explicitly say GPU execution was not verified.
