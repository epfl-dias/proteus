import json
import logging


def convert_tuple_type(obj):
    return [{"relName": fix_rel_name(t["rel"]), "attrName": t["attr"]} for t in obj]


def convert_operator(obj):
    op = obj["operator"]
    converters = {
        "agg": convert_agg,
        "join": convert_join,
        "projection": convert_projection,
        "select": convert_selection,
        "scan": convert_scan,
        "sort": convert_sort,
        "default": convert_default
    }
    if op in converters:
        conv = converters[op](obj)
    else:
        conv = converters["default"](obj)
    conv["output"] = convert_tuple_type(obj["tupleType"])
    return conv


def convert_default(obj):
    logging.warning("No converter for operator: " + obj["operator"])
    possible_inputs = ["input", "left", "right"]
    for inp in possible_inputs:
        if inp in obj:
            obj[inp] = convert_operator(obj[inp])
    return obj


def convert_expr_default(obj, parent):
    logging.warning("No converter for expression: " + obj["expression"])
    if "e" in obj:
        obj["e"] = convert_expression("e", obj)
    if "type" in obj:
        obj["type"] = convert_type(obj["type"])
    return obj


convert_binexpr_type = {
    "mult": "multiply"
}


def convert_expr_binary(obj, parent):
    if obj["expression"] in convert_binexpr_type:
        obj["expression"] = convert_binexpr_type[obj["expression"]]
    if "type" in obj:
        obj["type"] = convert_type(obj["type"])
    obj["left"] = convert_expression(obj["left"], obj)
    obj["right"] = convert_expression(obj["right"], obj)
    return obj


conv_type = {
    "INTEGER": "int",
    "BIGINT": "int64",
    "BOOLEAN": "bool",
    "VARCHAR": "dstring"
}


def convert_type(type):
    if type.startswith("CHAR("):
        return "dstring"
    return conv_type[type]


def convert_expr_int(obj, parent):
    return {"expression": "int", "v": int(obj["v"])}


def convert_expr_varchar(obj, parent):
    assert(parent is not None)
    assert("depends_on" in parent)
    assert(len(parent["depends_on"]) == 1)
    string_attr = parent["depends_on"][0]
    path = string_attr["relName"] + "." + string_attr["attrName"] + ".dict"
    return {
        "expression": "dstring",
        "v": obj["v"][1:-1],
        "dict": {
            "path": path
        }
    }


def convert_expression(obj, parent=None):
    if "rel" in obj and "attr" in obj:
        # print(obj)
        return {
                    "expression": "recordProjection",
                    "e": {
                        "expression": "argument",
                        "argNo": -1,
                        "type": {
                            "type": "record",
                            "relName": fix_rel_name(obj["rel"])
                        },
                        "attributes": [{
                            "relName": fix_rel_name(obj["rel"]),
                            "attrName": obj["attr"]
                        }]
                    },
                    "attribute": {
                        "relName": fix_rel_name(obj["rel"]),
                        "attrName": obj["attr"]
                    }
                }
    if "depends_on" in obj:
        dep_new = []
        for d in obj["depends_on"]:
            dep_new.append({
                "relName": fix_rel_name(d["rel"]),
                "attrName": d["attr"]
            })
        obj["depends_on"] = dep_new
    # obj["e"] = convert_expression(obj["e"])
    converters = {
        "mult": convert_expr_binary,
        "eq": convert_expr_binary,
        "le": convert_expr_binary,
        "lt": convert_expr_binary,
        "gt": convert_expr_binary,
        "ge": convert_expr_binary,
        "or": convert_expr_binary,
        "and": convert_expr_binary,
        "sub": convert_expr_binary,
        "INTEGER": convert_expr_int,
        "VARCHAR": convert_expr_varchar,
        "default": convert_expr_default
    }
    if obj["expression"].startswith("CHAR("):
        return convert_expr_varchar(obj, parent)
    if obj["expression"] in converters:
        return converters[obj["expression"]](obj, parent)
    return converters["default"](obj, parent)


def convert_scan(obj):
    conv = {"operator": "scan"}
    conv["plugin"] = {"type": "block", "name": fix_rel_name(obj["name"])}
    conv["plugin"]["projections"] = [{"relName": fix_rel_name(t["rel"]),
                                      "attrName": t["attr"]}
                                     for t in obj["tupleType"]]
    return conv


def convert_sort(obj):
    conv = {}
    conv["operator"] = "sort"
    exp = []
    d = []
    for (arg, t) in zip(obj["args"], obj["tupleType"]):
        d.append(arg["direction"])
        e = convert_expression(arg["expression"])
        # e["type"] = {
        #     "type": convert_type(t["type"]]
        # }
        e["register_as"] = {
            "type": {
                "type": convert_type(t["type"])
            },
            "attrNo": -1,
            "relName": fix_rel_name(t["rel"]),
            "attrName": t["attr"]
        }
        exp.append(e)
    conv["e"] = exp
    conv["direction"] = d
    if "input" in obj:
        conv["input"] = convert_operator(obj["input"])
    return conv


rel_names = {
    "dates": "inputs/ssbm100/date.csv",
    "lineorder": "inputs/ssbm100/lineorder.csv",
    "supplier": "inputs/ssbm100/supplier.csv",
    "part": "inputs/ssbm100/part.csv",
    "customer": "inputs/ssbm100/customer.csv"
}


def fix_rel_name(obj):
    if obj in rel_names:
        return rel_names[obj]
    return obj


def convert_agg(obj):
    conv = {}
    conv["operator"] = "reduce"
    conv["p"] = {"expression": "bool", "v": True}
    exp = []
    acc = []
    for (agg, t) in zip(obj["aggs"], obj["tupleType"]):
        if agg["op"] == "cnt":
            acc.append("sum")
            e = {
                "expression": convert_type(t["type"]),
                "v": 1
            }
        else:
            acc.append(agg["op"])
            e = convert_expression(agg["e"])
            # e["type"] = {
            #     "type": convert_type(t["type"]]
            # }
        e["register_as"] = {
            "type": {
                "type": convert_type(t["type"])
            },
            "attrNo": -1,
            "relName": fix_rel_name(t["rel"]),
            "attrName": t["attr"]
        }
        exp.append(e)
    conv["e"] = exp
    conv["accumulator"] = acc
    if "input" in obj:
        conv["input"] = convert_operator(obj["input"])
    return conv


def convert_projection(obj):
    conv = {}
    conv["operator"] = "project"
    conv["relName"] = fix_rel_name(obj["tupleType"][0]["rel"])
    conv["e"] = []
    assert(len(obj["tupleType"]) == len(obj["e"]))
    for (t, e) in zip(obj["tupleType"], obj["e"]):
        exp = convert_expression(e)
        exp["register_as"] = {
            "type": {
                "type": convert_type(t["type"])
            },
            "attrNo": -1,
            "relName": fix_rel_name(t["rel"]),
            "attrName": t["attr"]
        }
        conv["e"].append(exp)
    conv["input"] = convert_operator(obj["input"])
    return conv


def convert_selection(obj):
    conv = {}
    conv["operator"] = "select"
    conv["p"] = convert_expression(obj["cond"])
    conv["input"] = convert_operator(obj["input"])
    return conv


type_size = {
    "bool": 1,  # TODO: check this
    "int": 32,
    "int64": 64,
    "dstring": 32
}


def convert_join(obj):  # FIMXE: for now, right-left is in reverse, for the star schema
    conv = {}
    conv["operator"] = "hashjoin-chained"
    conv["hash_bits"] = 14                     # FIXME: make more adaptive
    conv["maxBuildInputSize"] = 1024*1024*128  # FIXME: requires upper limit!
    if (obj["cond"]["expression"] != "eq"):
        print(obj["cond"]["expression"])
        assert(False)
    conv["build_k"] = convert_expression(obj["cond"]["right"])
    conv["build_k"]["type"] = {
        "type": convert_type(obj["cond"]["right"]["type"])
    }
    conv["probe_k"] = convert_expression(obj["cond"]["left"])
    conv["probe_k"]["type"] = {
        "type": convert_type(obj["cond"]["left"]["type"])
    }
    build_e = []
    leftTuple = obj["right"]["tupleType"]
    leftAttr = obj["cond"]["right"].get("attr", None)
    build_packet = 1
    conv["build_w"] = [32 + type_size[conv["build_k"]["type"]["type"]]]
    for t in leftTuple:
        if t["attr"] != leftAttr:
            for x in obj["tupleType"]:
                if x["attr"] == t["attr"]:
                    e = convert_expression(t)
                    out_t = x
                    out_type = convert_type(out_t["type"])
                    e["register_as"] = {
                        "type": {
                            "type": out_type
                        },
                        "attrNo": -1,
                        "relName": fix_rel_name(out_t["rel"]),
                        "attrName": out_t["attr"]
                    }
                    conv["build_w"].append(type_size[out_type])
                    be = {"e": e}
                    be["packet"] = build_packet
                    build_packet = build_packet + 1
                    be["offset"] = 0
                    build_e.append(be)
                    break
        else:
            for x in obj["tupleType"]:
                if x["attr"] == leftAttr:
                    out_t = x
                    out_type = convert_type(out_t["type"])
                    conv["build_k"]["register_as"] = {
                        "type": {
                            "type": out_type
                        },
                        "attrNo": -1,
                        "relName": fix_rel_name(out_t["rel"]),
                        "attrName": out_t["attr"]
                    }
                    break
    conv["build_e"] = build_e
    conv["build_input"] = convert_operator(obj["right"])
    probe_e = []
    rightTuple = obj["left"]["tupleType"]
    rightAttr = obj["cond"]["left"].get("attr", None)
    probe_packet = 1
    conv["probe_w"] = [32 + type_size[conv["probe_k"]["type"]["type"]]]
    for t in rightTuple:
        if t["attr"] != rightAttr:
            for x in obj["tupleType"]:
                if x["attr"] == t["attr"]:
                    e = convert_expression(t)
                    out_t = x
                    out_type = convert_type(out_t["type"])
                    e["register_as"] = {
                        "type": {
                            "type": out_type
                        },
                        "attrNo": -1,
                        "relName": fix_rel_name(out_t["rel"]),
                        "attrName": out_t["attr"]
                    }
                    conv["probe_w"].append(type_size[out_type])
                    pe = {"e": e}
                    pe["packet"] = probe_packet
                    probe_packet = probe_packet + 1
                    pe["offset"] = 0
                    probe_e.append(pe)
                    break
        else:
            for x in obj["tupleType"]:
                if x["attr"] == rightAttr:
                    out_t = x
                    out_type = convert_type(out_t["type"])
                    conv["probe_k"]["register_as"] = {
                        "type": {
                            "type": out_type
                        },
                        "attrNo": -1,
                        "relName": fix_rel_name(out_t["rel"]),
                        "attrName": out_t["attr"]
                    }
                    break
    conv["probe_e"] = probe_e
    conv["probe_input"] = convert_operator(obj["left"])
    return conv


def translate_plan(obj):
    out = convert_operator(obj)
    conv = {"operator": "print", "e": []}
    for e in obj["tupleType"]:
        conv["e"].append(convert_expression(e))
    conv["input"] = out
    conv["output"] = []
    return conv


if __name__ == "__main__":
    plan = json.load(open("projected_plan.json"))
    out = translate_plan(plan)
    # print(json.dumps(plan, sort_keys=False, indent=4));
    print(json.dumps(out, indent=4))
    # print(json.dumps(plan, indent=4))
