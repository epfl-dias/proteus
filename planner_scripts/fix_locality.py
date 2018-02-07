import json
import logging


input_keys = ["input", "build_input", "probe_input"]
block_ops = ["scan"]
tuple_ops = ["reduce", "hashjoin-chained", "select", "project"]


def fixlocality_operator(obj, target=None):
    if obj["operator"] == "block-to-tuples":
        assert(target is None)
        if not obj["gpu"]:  # cpu
            mem_mov = {"operator": "mem-move-device"}
            mem_mov["to_cpu"] = True
            mem_mov["projections"] = []
            for t in obj["output"]:
                mem_mov["projections"].append({
                    "relName": t["relName"],
                    "attrName": t["attrName"],
                    "isBlock": True
                })
            mem_mov["input"] = fixlocality_operator(obj["input"], None)
            mem_mov["output"] = obj["output"]
            mem_mov["blockwise"] = True
            mem_mov["gpu"] = False
            obj["input"] = mem_mov
        else:
            obj["input"] = fixlocality_operator(obj["input"], "gpu")
    elif target:
        if obj["operator"] == "cpu-to-gpu" or not obj["gpu"]:
            for inp in input_keys:
                if inp in obj:
                    mem_mov = {"operator": "mem-move-device"}
                    mem_mov["to_cpu"] = (target != "gpu")
                    mem_mov["projections"] = []
                    for t in obj["output"]:
                        mem_mov["projections"].append({
                            "relName": t["relName"],
                            "attrName": t["attrName"],
                            "isBlock": True
                        })
                    mem_mov["input"] = fixlocality_operator(obj[inp], None)
                    mem_mov["output"] = obj["output"]
                    mem_mov["blockwise"] = True
                    mem_mov["gpu"] = False
                    obj[inp] = mem_mov
    else:
        for inp in input_keys:
            if inp in obj:
                obj[inp] = fixlocality_operator(obj[inp], target)
    return obj


if __name__ == "__main__":
    plan = json.load(open("deviceaware_flowaware_plan.json"))
    out = fixlocality_operator(plan)
    # print(json.dumps(plan, sort_keys=False, indent=4));
    print(json.dumps(out, indent=4))
    # print(json.dumps(plan, indent=4))
