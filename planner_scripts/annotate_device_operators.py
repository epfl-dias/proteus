import json
import logging


input_keys = ["input", "build_input", "probe_input"]
block_ops = ["scan"]
tuple_ops = ["reduce", "hashjoin-chained", "select", "project"]
devices = ["cpu", "gpu"]


def annotate_device_operator(obj, coming_from_gpu=False):
    gpu_op = coming_from_gpu
    if "jumpTo" in obj:
        assert(obj["jumpTo"] in devices)
        jtgpu = (obj["jumpTo"] == "gpu")
        assert(jtgpu == coming_from_gpu)
        # del obj["jumpTo"]
        gpu_op = not jtgpu
    obj["gpu"] = gpu_op
    ok = not gpu_op  # leaf nodes should always be CPU ops
    for inp in input_keys:
        if inp in obj:
            annotate_device_operator(obj[inp], gpu_op)
            ok = True
    assert(ok)
    return obj


if __name__ == "__main__":
    plan = json.load(open("device_jumps_annotated_plan.json"))
    out = annotate_device_operator(plan)
    # print(json.dumps(plan, sort_keys=False, indent=4));
    print(json.dumps(out, indent=4))
    # print(json.dumps(plan, indent=4))
