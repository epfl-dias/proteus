import subprocess
import os
import json
import re

do_execute = True
do_clean = True

dates = r"""d_datekey:int,d_date:string,d_dayofweek:string,d_month:string,d_year:int,d_yearmonthnum:int,d_yearmonth:string,d_daynuminweek:int,d_daynuminmonth:int,d_daynuminyear:int,d_monthnuminyear:int,d_weeknuminyear:int,d_sellingseason:string,d_lastdayinweekfl:boolean,d_lastdayinmonthfl:boolean,d_holidayfl:boolean,d_weekdayfl:boolean"""
customer = r"""c_custkey:int,c_name:string,c_address:string,c_city:string,c_nation:string,c_region:string,c_phone:string,c_mktsegment:string"""
supplier = r"""s_suppkey:int,s_name:string,s_address:string,s_city:string,s_nation:string,s_region:string,s_phone:string"""
part = r"""p_partkey:int,p_name:string,p_mfgr:string,p_category:string,p_brand1:string,p_color:string,p_type:string,p_size:int,p_container:string"""
lineorder = r"""lo_orderkey:int,lo_linenumber:int,lo_custkey:int,lo_partkey:int,lo_suppkey:int,lo_orderdate:int,lo_orderpriority:string,lo_shippriority:string,lo_quantity:int,lo_extendedprice:int,lo_ordtotalprice:int,lo_discount:int,lo_revenue:int,lo_supplycost:int,lo_tax:int,lo_commitdate:int,lo_shipmode:string"""

tpch = {
    "part"      : r"""
    p_partkey       int     PRIMARY KEY,
    p_name          string,
    p_mfgr          string,
    p_brand         string,
    p_type          string,
    p_size          int,
    p_container     string,
    p_retailprice   float,
    p_comment       string
""",
    "supplier"  : r"""
    s_suppkey       int     PRIMARY KEY,
    s_name          string,
    s_address       string,
    s_nationkey     int     FOREIGN KEY REFERENCES nation  (n_nationkey),
    s_phone         string,
    s_acctbal       float,
    s_comment       string
""",
    "partsupp"  : r"""
    ps_partkey      int     FOREIGN KEY REFERENCES part    (p_partkey ),
    ps_suppkey      int     FOREIGN KEY REFERENCES supplier(s_suppkey ),
    ps_availqty     int,
    ps_supplycost   float,
    ps_comment      string
""",
    "customer"  : r"""
    c_custkey       int     PRIMARY KEY,
    c_name          string,
    c_address       string,
    c_nationkey     int     FOREIGN KEY REFERENCES nation  (n_nationkey),
    c_phone         string,
    c_acctbal       float,
    c_mktsegment    string,
    c_comment       string
""",
    "orders"    : r"""
    o_orderkey      int     PRIMARY KEY,
    o_custkey       int     FOREIGN KEY REFERENCES customer(c_custkey),
    o_orderstatus   string,
    o_totalprice    float,
    o_orderdate     date,
    o_orderpriority string,
    o_clerk         string,
    o_shippriority  int,
    o_comment       string
""",
    "lineitem"  : r"""
    l_orderkey      int     FOREIGN KEY REFERENCES orders  (o_orderkey),
    l_partkey       int     FOREIGN KEY REFERENCES part    (p_partkey ),
    l_suppkey       int     FOREIGN KEY REFERENCES supplier(s_suppkey ),
    l_linenumber    int,
    l_quantity      float,
    l_extendedprice float,
    l_discount      float,
    l_tax           float,
    l_returnflag    string,
    l_linestatus    string,
    l_shipdate      date,
    l_commitdate    date,
    l_receiptdate   date,
    l_shipinstruct  string,
    l_shipmode      string,
    l_comment       string
""",
    "nation"    : r"""
    n_nationkey     int     PRIMARY KEY,
    n_name          string,
    n_regionkey     int,
    n_comment       string
""",
    "region"    : r"""
    r_regionkey     int     PRIMARY KEY,
    r_name          string,
    r_comment       string
"""
}

def run(cmd):
    print(cmd)
    if do_execute:
        process = subprocess.Popen(["time", "bash", "-c", cmd], stdout=subprocess.PIPE)
        output, error = process.communicate()
        if output:
            print("output: " + output)
        if error:
            print("error : " + error)


def clean_up(file):
    if do_clean:
        print("rm " + file + " # Deleting file: " + file)
        os.remove(file)
    else:
        print("# TODO: delete file: " + file)


def prepare_attr(relName, attrName, attrIndex, type, relPathName):
    # extract column
    argNo = attrIndex + 1
    run(r"""# prepare """ + relName + "." + attrName)
    # if (attrName in ["lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey", "lo_suppkey", "lo_orderdate"]): return;
    columnFileTxt = attrName + ".csv"
    columnFileBin = relName + ".csv." + attrName
    # extract_cmd = "cat " + relName + ".tbl | cut -d '|' -f" + str(attrIndex + 1) + " > " + columnFileTxt
    # run(extract_cmd)
    # convert to binary based on type
    pelago_type = type
    if type == "int" or type == "boolean": # FIXME: do we handle bools as ints ? should we do something better?
        # for ints, just parse it
        # extract_cmd = "cat " + relName + ".tbl | cut -d '|' -f" + str(attrIndex + 1) + " > " + columnFileTxt
        # run(extract_cmd)
        # parse_cmd = r"""perl -pe '$_=pack"l",$_' < """ + columnFileTxt + r""" > """ + columnFileBin
        parse_cmd = "cat " + relName + ".tbl | cut -d '|' -f" + str(argNo) + " | " + r"""perl -pe '$_=pack"l",$_' > """ + columnFileBin
        run(parse_cmd)
        pelago_type = "int" # FIXME: booleans should NOT be considered ints
    elif type == "string":
        extract_cmd = "cat " + relName + ".tbl | cut -d '|' -f" + str(argNo) + " > " + columnFileTxt
        run(extract_cmd)
        columnFileDict = columnFileBin + ".dict"
        columnFileEnc = columnFileTxt + ".encoded"
        build_dict_cmd = r"""cat """ + columnFileTxt + r""" | sort -S1G --parallel=24 -u | uniq | awk '{printf("%s:%d\n", $0, NR-1)}' > """ + columnFileDict
        run(build_dict_cmd)
        encode_cmd = r"""awk -F: 'FNR==NR {dict[$1]=$2; next} {$1=($1 in dict) ? dict[$1] : $1}1' """ + columnFileDict + r""" """ + columnFileTxt + r""" > """ + columnFileEnc
        run(encode_cmd)
        parse_cmd = r"""perl -pe '$_=pack"l",$_' < """ + columnFileEnc + r""" > """ + columnFileBin
        run(parse_cmd)
        clean_up(columnFileEnc)
        clean_up(columnFileTxt)
        pelago_type = "dstring"
    elif type == "date":
        parse_cmd = "cat " + relName + ".tbl | cut -d '|' -f" + str(argNo) + " | " + r"""perl -MTime::Piece -pE '$_=pack("q",Time::Piece->strptime($_, "%Y-%m-%d\n")->epoch * 1000)' > """ + columnFileBin
        run(parse_cmd)
    elif type == "float":
        parse_cmd = "cat " + relName + ".tbl | cut -d '|' -f" + str(argNo) + " | " + r"""perl -pe '$_=pack"d",$_' > """ + columnFileBin
        run(parse_cmd)
    else:
        assert(False)  # Unknown type!
    return {
        "type": {
            "type": pelago_type
        },
        "relName": relPathName,
        "attrName": attrName,
        "attrNo": argNo
    }

def build_relname(schemaName, relName):
    return schemaName + "_" + relName

def create_constraint(attrName, s, get_relname):
    cs = []
    s = " " + ' '.join(s.split()).lower() + " " #get rid of extra whitespaces
    if " primary key " in s:
        cs.append({
            "type": "primary_key",
            "columns": [attrName]
        })
    if " unique " in s:
        cs.append({
            "type": "unique",
            "columns": [attrName]
        })
    if " foreign key references " in s:
        assert(s.count(" foreign key references ") == 1)
        m = re.match(r""".*foreign key references\s+(\w+)\s*\(\s*(\w+)\s*\).*""", s)
        if not m:
            print(s)
        assert(m)
        cs.append({
            "type": "foreign_key",
            "referencedTable": get_relname(m.group(1)),
            "references": [{
                "referee": attrName,
                "referred": m.group(2)
            }]
        })
    return cs

def prepare_rel(relName, namespace, relFolder, schemaName):
    schema = namespace[relName]
    columns = [attr.split(None, 2) for attr in schema.split(",")]
    relPathName = relFolder + "/" + schemaName + "/" + relName + ".csv"
    with open(relName + ".tbl") as f:
        linehint = sum(1 for _ in f)
    
    attrs = []
    constraints = []
    for (ind, c) in enumerate(columns):
        attrs.append(prepare_attr(relName, c[0], ind, c[1], relPathName))
        if len(c) > 2 and len(c[2]) > 0:
            constraints.extend(create_constraint(c[0], c[2], lambda x: build_relname(schemaName, x)))
    
    return {
        build_relname(schemaName, relName): {
            "path": relPathName,
            "type": {
                "type": "bag",
                "inner": {
                    "type": "record",
                    "attributes": attrs
                }
            },
            "plugin": {
                "type": "block",
                "linehint": linehint
            },
            "constraints": constraints
        }
    }


if __name__ == '__main__':
    relFolder  = "inputs"
    schemaName = "tpch100"
    
    namespace  = tpch
    catalog    = {}
    # print(json.dumps(prepare_rel("date", dates), indent = 4))
    # print(json.dumps(prepare_rel("customer", customer), indent = 4))
    # print(json.dumps(prepare_rel("supplier", supplier), indent = 4))
    # print(json.dumps(prepare_rel("part", part), indent = 4))
    # print(json.dumps(prepare_rel("lineorder", lineorder), indent = 4))
    
    for name in namespace:
        catalog.update(prepare_rel(name, namespace, relFolder, schemaName))
    
    with open('catalog.json', 'w') as out:
        out.write(json.dumps(catalog, indent = 4))
        out.write('\n')

