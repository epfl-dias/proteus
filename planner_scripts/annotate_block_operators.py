import json
import logging


input_keys = ["input", "build_input", "probe_input"]
block_ops = ["scan", "sort"]
tuple_ops = ["reduce", "hashjoin-chained", "select", "project", "print"]


def annotate_block_operator(obj):
    for inp in input_keys:
        if inp in obj:
            annotate_block_operator(obj[inp])
    if obj["operator"] in tuple_ops:
        obj["blockwise"] = False
    elif obj["operator"] in block_ops:
        obj["blockwise"] = True
    else:
        logging.error("Unknown operator: " + obj["operator"])
        assert(False)
    return obj


if __name__ == "__main__":
    plan = json.load(open("translated_plan.json"))
    out = annotate_block_operator(plan)
    # print(json.dumps(plan, sort_keys=False, indent=4));
    print(json.dumps(out, indent=4))
    # print(json.dumps(plan, indent=4))
