import json
import logging


input_keys = ["input", "build_input", "probe_input"]
block_ops = ["scan"]
tuple_ops = ["reduce", "hashjoin-chained", "select", "project", "print"]


def annotate_device_jumps_operator(obj, force_jump_to_cpu=False):
    is_print = (obj["operator"] == "print")
    for inp in input_keys:
        if inp in obj:
            annotate_device_jumps_operator(obj[inp], is_print)
    if obj["operator"] == "scan":
        if not force_jump_to_cpu:
            obj["jumpTo"] = "gpu"
    elif force_jump_to_cpu:
        obj["jumpTo"] = "cpu"
    return obj


if __name__ == "__main__":
    plan = json.load(open("flow_annotated_plan.json"))
    out = annotate_device_jumps_operator(plan)
    # print(json.dumps(plan, sort_keys=False, indent=4));
    print(json.dumps(out, indent=4))
    # print(json.dumps(plan, indent=4))
