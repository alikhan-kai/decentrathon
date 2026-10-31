"""Compare parent-produced setup_probe logs; never loads or runs a model."""
import argparse
import json
from pathlib import Path

def read(path):
    prefix = "STARTUP_SETUP_DIAGNOSTIC "
    matches = [json.loads(line[len(prefix):]) for line in Path(path).read_text().splitlines()
               if line.startswith(prefix)]
    assert len(matches) == 1, (path, "expected one diagnostic record")
    return matches[0]


def normalized_placement(record):
    mapping = record["hf_device_map"]
    if mapping is None:
        # Single-device Transformers setup need not retain hf_device_map.
        # Use the captured actual placement, never infer missing tensor devices.
        actual = {t["name"]: t["device"] for t in record["tensors"]}
        assert actual and len(actual) == len(record["tensors"])
        assert set(actual.values()) == {"cuda:0"}, "unverified absent-map placement"
        return actual
    assert isinstance(mapping, dict) and mapping, "missing placement map"
    def device(value):
        if type(value) is int:
            return "cuda:" + str(value)
        return str(value)
    output = {}
    for tensor in record["tensors"]:
        name = tensor["name"]
        keys = [key for key in mapping if key == "" or name == key or name.startswith(key + ".")]
        assert keys, ("unmapped tensor", name)
        output[name] = device(mapping[max(keys, key=len)])
    return output


def compare(a, b):
    assert a["variant"] == "control" and b["variant"] in ("device", "map", "both")
    for record in (a, b):
        assert record["setup_completed"] and record["full_run_completed"]
        assert record["exit_status"] == 0 and record["completed_steps"] == 64
        assert record["training_completed"] and record["training_calls"] == 1
        assert record["hook_evidence"] == "counts_and_qualified_type_only"
    for key in ("tensors", "quantization_config", "hooks"):
        assert a[key] == b[key], "setup metadata mismatch: " + key
    assert normalized_placement(a) == normalized_placement(b), "hf_device_map placement mismatch"
    def count(record, name):
        value = record["components_inclusive"].get(name, {}).get("calls", 0)
        assert type(value) is int and value >= 0
        return value
    for record in (a, b):
        device = record["variant"] in ("device", "both")
        fast_map = record["variant"] in ("map", "both")
        for name in ("infer_auto_device_map", "balanced_memory"):
            assert (count(record, name) == 0) == device, "device ablation contaminated/ineligible: " + name
        assert count(record, "soup_device_resolver") > 0, "no original resolver captured"
        assert (count(record, "normalized_map_core_inclusive") > 0) == fast_map, "map ablation contaminated/ineligible"
        assert count(record, "text_map_total") > 0, "no text mapping captured"
        if fast_map:
            assert count(record, "normalized_map_core_inclusive") == count(record, "text_map_total"), "partial map eligibility"
    assert count(a, "text_map_total") == count(b, "text_map_total")
    if b["variant"] == "map":
        for name in ("infer_auto_device_map", "balanced_memory", "soup_device_resolver"):
            assert count(a, name) == count(b, name), "map-only altered device call matrix"
    return {"setup_metadata_equal": True, "variant": b["variant"],
            "placement_evidence": "captured tensor devices when hf_device_map is absent",
            "hook_evidence": "count/type metadata parity only; callable behavior NOT established",
            "full_64_step_runs_completed": True,
            "control_times": a["components_inclusive"], "candidate_times": b["components_inclusive"],
            "remaining": "hook behavior, full outputs/Rewind/numerics and uninstrumented official T4 runs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("candidate")
    args = parser.parse_args()
    a, b = read(args.control), read(args.candidate)
    print(json.dumps(compare(a, b), indent=2))

if __name__ == "__main__":
    main()
