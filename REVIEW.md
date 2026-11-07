# Publication review

The review on 2026-11-06 and 2026-11-07 checked the report against archived inputs, code and outputs. The following results were re-evaluated after the initial study. The review and re-execution were performed with Codex.

## Inference record audit

The read-only verifier reconstructs all 360 output sequences, accepted prefixes, discarded drafts, first-rejection positions and 228 summary fields. Every archived request matches its saved ordinary greedy reference. There are six independent diagnostic prompts and five timing repeats per configuration. The request-history adaptive heuristic takes longer than fixed budget eight in all five paired comparisons. The median increase is 2.711 percent.

The scripts use a single-chain trigram proposer with SmolLM2 135M on CPU. This is a controlled diagnostic implementation. The observed batch contains zero EOS cases. The two fixed budgets were specified directly and were not selected using a development set. These details limit conclusions about new workloads and stronger controllers.

The machine-readable check is in `verification/inference_archived_check.json`. The helper checks saved records without executing the target model. A separate short execution checks the publication candidate's model entry point.

## Feedback experiment reproduction

A fresh full run repeated training and evaluation, followed by saved-array analysis. All 120 summary rows, 7,200 trajectory records and 120 model records exactly match the archived result files. The SHA256 hashes of `summary.csv`, `trajectory_metrics.csv` and `models.json` are identical. The fresh array audit covers 600 test files, 90 training files and 72,000 metric values.

The reproduction used Python 3.12.14 and NumPy 2.4.6. The original used Python 3.12.2 and the same NumPy version. Details are in `verification/driving_reproduction_check.json`.

The most informative mechanism is feedback identifiability. Exactly centered state data leave two feedback parameters unconstrained. Under slightly excited noiseless data, ordinary BC recovers the analytical expert. A 300 ms action queue destabilizes even that expert in this fixed system. These results motivate more precise studies of state coverage and timing.

## Sampling function rerun

The pinned upstream utility was invoked again with 30,000 samples per candidate ordering. The resulting probabilities reproduce the original counts. The independent calculation uses a target probability 0.8 and derives 0.928 or 0.768 under the inspected fallback and selection rules. The result applies to the specified function, constructed branches and upstream entry.

Details and hashes are in `verification/sampling_audit_rerun.json`.

## Next experiment design

For inference, select the fixed budget on a separate development workload, preserve the proposer, and compare a controller that accounts for measured time against the acceptance heuristic. Held-out task diversity and timing boundaries need to be specified before drawing conclusions.

For driving, use identical roads and fixed policies when comparing delay conditions. Record observation time, proposed-action time and executed-action time. A visual policy and a reactive simulator would add a materially different evaluation level after this low-dimensional mechanism check.

The technical report retains the outstanding access gap for the formal Adaptix article. Mechanism review uses the specified preprint and pinned code. Publication and application descriptions should retain the execution provenance in `PROVENANCE.md`.
