import json
import os
import sys


def get_inputs(obj):
    possible_inputs = ["input", "build_input", "probe_input"]
    ins = []
    if obj["operator"] == "scan":
        for x in obj["plugin"]["projections"]:
            ins.append(x["relName"] + "." + x["attrName"])
    for inkey in possible_inputs:
        if inkey in obj:
            ins += get_inputs(obj[inkey])
    return ins


def createTest(name):
    test_name = os.path.basename(name)
    if test_name.endswith(".json"):
        test_name = test_name[0:-5]
    test_name = os.path.basename(test_name).replace('.', '_')
    print(r"""
TEST_F(MultiGPUTest, """ + test_name + r""") {
    auto load = [](string filename){
        StorageManager::load(filename, PINNED);
    };
""")
    for f in get_inputs(json.load(open(name))):
        print('    load("' + f + '");')
    print(r'''
    const char *testLabel = "''' + test_name + r'''";
    const char *planPath  = "inputs/plans/''' + name + r'''";

    runAndVerify(testLabel, planPath);
}''')


if __name__ == "__main__":
    createTest(sys.argv[1])
