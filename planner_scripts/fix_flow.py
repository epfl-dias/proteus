import json
import logging


input_keys = ["input", "build_input", "probe_input"]
block_ops = ["scan"]
tuple_ops = ["reduce", "hashjoin-chained", "select", "project"]


def flowaware_operator(obj):
    for inp in input_keys:
        if inp in obj:
            inpobj = obj[inp]
            if "blockwise" not in obj:
                continue
            if "blockwise" not in inpobj:
                continue
            if obj["blockwise"] != inpobj["blockwise"]:
                obj[inp] = {}
                if obj["blockwise"]:
                    obj[inp]["operator"] = "tuples-to-block"
                else:
                    obj[inp]["operator"] = "block-to-tuples"
                projs = []
                for t in inpobj["output"]:
                    projs.append({
                        "relName": t["relName"],
                        "attrName": t["attrName"],
                        })
                obj[inp]["projections"] = projs
                obj[inp]["input"] = flowaware_operator(inpobj)
                obj[inp]["output"] = projs
                if "gpu" in obj:
                    obj[inp]["gpu"] = obj["gpu"]
            else:
                obj[inp] = flowaware_operator(inpobj)
    return obj


if __name__ == "__main__":
    plan = json.load(open("deviceaware_plan.json"))
    out = flowaware_operator(plan)
    # print(json.dumps(plan, sort_keys=False, indent=4));
    print(json.dumps(out, indent=4))
    # print(json.dumps(plan, indent=4))
