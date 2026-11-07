# Third party sources

The sampling audit loads the unmodified Adaptix utility at `sources/Adaptix/aded/model/utils.py`.
Its upstream commit is `241dad836fd71aa295a3967f7efe4cfcd2dd9609` and its SHA256 is `590910836297406869921120db2e06aee3f100bc1a85367a3bcf69fce90bb012`.
The upstream Apache 2.0 license is retained in `sources/Adaptix/LICENSE`.

The utility's original header credits [Medusa](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py).
Medusa also uses [Apache 2.0](https://github.com/FasterDecoding/Medusa/blob/main/LICENSE).
The header is preserved. The test supplies an empty `draftretriever` import stub because it never calls the Rust retriever.

The target model is [SmolLM2 135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
The model card specifies Apache 2.0. The download script retrieves the original public files and verifies the model weight SHA256.

Paper links and the inspected versions appear in `reports/evidence_matrix.md`.
The study discusses several published systems. Each experimental implementation in this repository is identified by its actual scope.

No additional license grant is currently specified for original study material. Upstream files retain their original terms.
