"""Verify saved inference records without loading a model or changing results."""

import argparse
import collections
import json
import math
import pathlib
import statistics
import sys


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected):
    if isinstance(actual, list):
        return (
            isinstance(expected, list)
            and len(actual) == len(expected)
            and all(same_number(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(actual, bool):
        return isinstance(expected, bool) and actual == expected
    if isinstance(actual, int):
        return not isinstance(expected, bool) and actual == expected
    return (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
    )


def request_totals(rows, repeats):
    return [
        sum(row["warm_request_s"] for row in rows if row["repeat"] == repeat)
        for repeat in repeats
    ]


def summary_for(rows, baseline, repeats):
    times = request_totals(rows, repeats)
    base_times = request_totals(baseline, repeats)
    ratios = [base / measured for base, measured in zip(base_times, times)]
    rounds = [item for row in rows for item in row["rounds"]]
    tokens = sum(row["output_tokens"] for row in rows)
    computed = {
        "requests": len(rows),
        "output_tokens": tokens,
        "median_total_s_per_repeat": statistics.median(times),
        "min_total_s": min(times),
        "max_total_s": max(times),
        "paired_speedups": ratios,
        "median_speedup": statistics.median(ratios),
        "accepted_per_round": sum(item["accepted"] for item in rounds) / len(rounds),
        "committed_per_round": tokens / len(rounds),
        "draft_tokens": sum(len(item["draft"]) for item in rounds),
        "accepted_draft_tokens": sum(item["accepted"] for item in rounds),
        "discarded_draft_tokens": sum(item["discarded"] for item in rounds),
        "failed_rounds": sum(item["failed"] for item in rounds),
        "failed_round_verify_s": sum(item["verify_s"] for item in rounds if item["failed"]),
        "all_consistent": all(row["consistent"] for row in rows),
    }
    for field in ("verify_s", "draft_s", "commit_s", "rollback_s"):
        computed[field] = sum(item[field] for item in rounds)
    return computed


def find_eos_id(results, requested, metadata):
    if requested is not None:
        return requested
    if "eos_token_id" in metadata:
        return metadata["eos_token_id"]
    model_config = results.parent.parent / "sources" / "SmolLM2-135M" / "config.json"
    if model_config.is_file():
        return read_json(model_config).get("eos_token_id")
    return None


def verify(results, eos_token_id=None):
    metadata = read_json(results / "metadata.json")
    references = read_json(results / "greedy_references.json")
    saved_summary = read_json(results / "summary.json")
    rows = [
        json.loads(line)
        for line in (results / "records.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    check(bool(rows), "No saved inference requests.")
    repeats = list(range(metadata["repeats"]))
    prompts = {item[0] for item in metadata["prompts"]}
    configurations = {(item["scope"], item["method"]) for item in saved_summary}
    check(len(configurations) == len(saved_summary), "Duplicate summary configurations.")
    protocol_configurations = {
        (scope, method)
        for scope in ("prompt_only", "request", "cross")
        for method in ("ar", "fixed4", "fixed8", "adaptive")
    }
    check(configurations == protocol_configurations, "Configurations differ from the measured experiment protocol.")
    check(set(references) == prompts, "Reference prompt IDs differ from metadata.")
    check(bool(configurations), "No method and history configurations in summary.")
    check(bool(repeats) and bool(prompts), "Metadata has no prompts or timing repeats.")
    counts = collections.Counter(
        (row["repeat"], row["prompt_id"], row["scope"], row["method"])
        for row in rows
    )
    expected = {
        (repeat, prompt, scope, method)
        for repeat in repeats
        for prompt in prompts
        for scope, method in configurations
    }
    check(set(counts) == expected, "Missing or unexpected request configuration.")
    check(all(count == 1 for count in counts.values()), "Duplicate request record.")
    eos = find_eos_id(results, eos_token_id, metadata)
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else None
    eos_requests = 0
    shortcuts = 0
    target_calls = 0
    for row in rows:
        label = f"{row['repeat']}/{row['prompt_id']}/{row['scope']}/{row['method']}"
        output = row["output_ids"]
        check(row["consistent"] is True, f"Saved consistency check failed in {label}.")
        check(output == references[row["prompt_id"]], f"Reference IDs differ in {label}.")
        check(len(output) == row["output_tokens"], f"Wrong output count in {label}.")
        check(bool(row["rounds"]), f"No generation rounds in {label}.")
        check(
            [token for item in row["rounds"] for token in item["committed"]] == output,
            f"Committed rounds do not reconstruct output in {label}.",
        )
        for field in ("warm_request_s", "prefill_s", "decode_s", "setup_s"):
            value = row[field]
            check(math.isfinite(value) and value >= 0, f"Invalid {field} in {label}.")
        check(row["warm_request_s"] > 0, f"Zero request time in {label}.")
        position = 0
        for item in row["rounds"]:
            accepted = item["accepted"]
            draft = item["draft"]
            commit = item["committed"]
            check(isinstance(accepted, int) and 0 <= accepted <= len(draft), f"Invalid acceptance count in {label}.")
            check(len(commit) == 1 + accepted, f"Wrong committed prefix length in {label}.")
            check(commit[1:] == draft[:accepted], f"Committed prefix differs from draft in {label}.")
            check(len(draft) == accepted + item["discarded"], f"Draft accounting differs in {label}.")
            check(item["failed"] == (accepted < len(draft)), f"Failure flag differs in {label}.")
            if item["failed"]:
                index = position + 1 + accepted
                check(index < len(output), f"First rejection has no reference token in {label}.")
                check(draft[accepted] != output[index], f"First rejection matches the reference in {label}.")
            for field in ("draft_s", "verify_s", "commit_s", "rollback_s", "round_s"):
                value = item[field]
                check(math.isfinite(value) and value >= 0, f"Invalid round {field} in {label}.")
            shortcuts += int(item["final_shortcut"])
            target_calls += int(not item["final_shortcut"])
            position += len(commit)
        if eos_ids is not None:
            eos_positions = [index for index, token in enumerate(output) if token in eos_ids]
            check(not eos_positions or eos_positions == [len(output) - 1], f"Tokens appear after EOS in {label}.")
            eos_requests += bool(eos_positions)
        check(len(output) <= metadata["max_new_tokens"], f"Length limit exceeded in {label}.")

    aggregated_fields = 0
    for saved in saved_summary:
        scope = saved["scope"]
        method = saved["method"]
        subset = [row for row in rows if row["scope"] == scope and row["method"] == method]
        baseline = [row for row in rows if row["scope"] == scope and row["method"] == "ar"]
        check(len(baseline) == len(prompts) * len(repeats), f"Missing AR baseline for {scope}.")
        for field, actual in summary_for(subset, baseline, repeats).items():
            check(same_number(actual, saved[field]), f"Summary mismatch for {scope}/{method}/{field}.")
            aggregated_fields += 1

    comparisons = {}
    contrast_specs = (
        ("adaptive_vs_fixed8_request", ("request", "adaptive"), ("request", "fixed8")),
        ("adaptive_vs_fixed8_cross", ("cross", "adaptive"), ("cross", "fixed8")),
        ("cross_vs_request_fixed8", ("cross", "fixed8"), ("request", "fixed8")),
    )
    for name, numerator, denominator in contrast_specs:
        if numerator not in configurations or denominator not in configurations:
            continue
        top = request_totals([row for row in rows if (row["scope"], row["method"]) == numerator], repeats)
        bottom = request_totals([row for row in rows if (row["scope"], row["method"]) == denominator], repeats)
        ratios = [a / b for a, b in zip(top, bottom)]
        comparisons[name] = {
            "time_ratios": ratios,
            "median_percent_change": 100 * (statistics.median(ratios) - 1),
            "all_greater_than_one": all(ratio > 1 for ratio in ratios),
        }
    saved_contrasts = results / "paired_contrasts.json"
    if saved_contrasts.is_file():
        for name, recorded in read_json(saved_contrasts).items():
            check(name in comparisons, f"Unexpected saved paired comparison {name}.")
            for field, actual in comparisons[name].items():
                check(same_number(actual, recorded[field]), f"Paired comparison mismatch for {name}/{field}.")

    return {
        "status": "passed",
        "requests": len(rows),
        "unique_prompts": len(prompts),
        "timing_repeats": len(repeats),
        "method_history_configurations": len(configurations),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "all_output_ids_equal_saved_references": True,
        "all_committed_prefixes_and_first_rejections_valid": True,
        "aggregate_fields_recomputed": aggregated_fields,
        "actual_decode_target_calls": target_calls,
        "prefill_calls": len(rows),
        "stop_coverage": {
            "eos_token_id": eos,
            "requests_with_eos": eos_requests if eos_ids is not None else None,
            "eos_coverage_known": eos_ids is not None,
            "final_shortcut_rounds": shortcuts,
            "requests_reaching_length_limit": sum(len(row["output_ids"]) == metadata["max_new_tokens"] for row in rows),
            "max_new_tokens": metadata["max_new_tokens"],
        },
        "paired_comparisons": comparisons,
        "interpretation": "Saved-record audit. Timing repeats reuse the same prompts. No model execution or general speedup claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=pathlib.Path)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        report = verify(args.results, args.eos_token_id)
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            # Exclusive creation protects measured and previously generated records.
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(text)
    except (ValueError, KeyError, OSError, TypeError, ZeroDivisionError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
