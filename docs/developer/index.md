# Developer notes

Material for working *on* `edmtn` rather than with it: the architecture
and its extension points, the testing contract, design decision ledgers
and dated benchmark records.

```{toctree}
:caption: Working on edmtn
:maxdepth: 1

architecture
testing
```

The design ledgers document decisions as they were made and are kept for
traceability; where a later change superseded one, the current code and the
{doc}`../api/index` win. The benchmarks record specific measurements — each
states its hardware, parameters and date — and are historical records, not
current performance guarantees.

```{toctree}
:caption: Design ledgers
:maxdepth: 1

../design/phase0-replatform-decisions
../design/multi-gpu-cuquantum-design
```

```{toctree}
:caption: Benchmarks
:maxdepth: 1

../benchmarks/cpu-vs-gpu-edm
../benchmarks/gpu-scaling-benchmark
```
