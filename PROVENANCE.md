# Research and execution provenance

The repository owner raised the initial cache-hit analogy and the question of how failed drafts affect net decoding time, and requested a study of physical-world AI with emphasis on end-to-end driving.

Codex produced the initial literature synthesis, implementation, experiment execution, and report draft in response to those questions. A subsequent Codex review checked the archived outputs and executed additional reproducibility checks. These automated activities are recorded as such. Personal contributions should be described according to actual participation.

The archived inference benchmark and driving simulation scripts are copied byte for byte. `provenance/source_snapshot.json` records their original hashes and the hashes of copied result files. New verification helpers are labeled separately.

The Chinese technical review is edited from the original research report. It retains technical analysis and measured results. Personal application and advisor-selection notes stay in the local research archive.

The experiments provide limited mechanism checks and baseline comparisons. The inference benchmark uses a deterministic single-chain trigram proposer on a CPU model. The lane experiment uses directly observed low-dimensional state, a linear policy, and an analytical expert. The sampling audit exercises a pinned upstream function with a constructed distribution. These scopes should be retained when discussing the work.
