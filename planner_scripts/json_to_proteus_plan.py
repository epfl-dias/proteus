import json
import logging
from filterColumns import get_required_input
from convert_plan import translate_plan
from annotate_block_operators import annotate_block_operator
from annotate_device_jumps_operators import annotate_device_jumps_operator
from annotate_device_operators import annotate_device_operator
from fix_device_flow import deviceaware_operator
from fix_locality import fixlocality_operator
from fix_flow import flowaware_operator


class plan:
    def __init__(self, str):
        self.p = json.load(open(str))

    def get_required_input(self):
        self.p = get_required_input(self.p)
        return self

    def translate_plan(self):
        self.p = translate_plan(self.p)
        return self

    def annotate_block_operator(self):
        self.p = annotate_block_operator(self.p)
        return self

    def annotate_device_jumps_operator(self):
        self.p = annotate_device_jumps_operator(self.p)
        return self

    def annotate_device_operator(self):
        self.p = annotate_device_operator(self.p)
        return self

    def deviceaware_operator(self):
        self.p = deviceaware_operator(self.p)
        return self

    def fixlocality_operator(self):
        self.p = fixlocality_operator(self.p)
        return self

    def flowaware_operator(self):
        self.p = flowaware_operator(self.p)
        return self

    def dump(self):
        return json.dumps(self.p, indent=4)


if __name__ == "__main__":
    out = plan("current.json")                                              \
              .get_required_input()                                         \
              .translate_plan()                                             \
              .annotate_block_operator()                                    \
              .annotate_device_jumps_operator()                             \
              .annotate_device_operator()                                   \
              .deviceaware_operator()                                       \
              .flowaware_operator()                                         \
              .fixlocality_operator()
    print(out.dump())
