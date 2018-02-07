import json


def get_dependencies(obj):
    if isinstance(obj, list):
        deps = []
        for x in obj:
            deps += get_dependencies(x)
        return deps
    if not isinstance(obj, dict):
        return []
    deps = []
    if "rel" in obj and "attr" in obj:
        deps += [{"rel": obj["rel"], "attr": obj["attr"]}]
    for key in obj:
        deps += get_dependencies(obj[key])
    obj["depends_on"] = deps
    return deps


def get_required_input(obj, output=None):
    possible_inputs = ["input", "left", "right"]
    deps = []
    if output:
        deps += output
    for key in obj:
        if key not in possible_inputs and key != "tupleType":
            deps += get_dependencies(obj[key])
    # deps = [tuple(x.items()) for x in deps]
    # print(attrs)
    # print(deps)
    if output:
        attrs = [x["attr"] for x in output]
        actual_output = []
        for out in obj["tupleType"]:
            if out["attr"] in attrs:
                actual_output.append({"rel": out["rel"],
                                      "attr": out["attr"],
                                      "type": out["type"]})
        # for projection, also delete expressions
        if obj["operator"] == "projection":
            actual_e = []
            for (out, e1) in zip(obj["tupleType"], obj["e"]):
                if out["attr"] in attrs:
                    actual_e.append(e1)
            obj["e"] = actual_e
    else:
        actual_output = obj["tupleType"]
    obj["tupleType"] = actual_output
    # print(actual_output)
    for inkey in possible_inputs:
        if inkey in obj:
            get_required_input(obj[inkey], deps)
    return obj


if __name__ == "__main__":
    plan = json.load(open("plan.json"))
    out = get_required_input(plan)
    # print(json.dumps(plan, sort_keys=False, indent=4));
    print(json.dumps(out, indent=4))
