import subprocess
import os
import json
import re
from multiprocessing.pool import ThreadPool
import argparse

pool = ThreadPool(processes=os.cpu_count())

do_execute = True
do_clean = True

ddl = {
"ssb": {
    "date"      : r"""
        d_datekey           int     PRIMARY KEY,
        d_date              string,
        d_dayofweek         string,
        d_month             string,
        d_year              int,
        d_yearmonthnum      int,
        d_yearmonth         string,
        d_daynuminweek      int,
        d_daynuminmonth     int,
        d_daynuminyear      int,
        d_monthnuminyear    int,
        d_weeknuminyear     int,
        d_sellingseason     string,
        d_lastdayinweekfl   boolean,
        d_lastdayinmonthfl  boolean,
        d_holidayfl         boolean,
        d_weekdayfl         boolean
    """,
    "customer"  : r"""
        c_custkey           int     PRIMARY KEY,
        c_name              string,
        c_address           string,
        c_city              string,
        c_nation            string,
        c_region            string,
        c_phone             string,
        c_mktsegment        string
    """,
    "supplier"  : r"""
        s_suppkey           int     PRIMARY KEY,
        s_name              string,
        s_address           string,
        s_city              string,
        s_nation            string,
        s_region            string,
        s_phone             string
    """,
    "part"      : r"""
        p_partkey           int     PRIMARY KEY,
        p_name              string,
        p_mfgr              string,
        p_category          string,
        p_brand1            string,
        p_color             string,
        p_type              string,
        p_size              int,
        p_container         string
    """,
    "lineorder" : r"""
        lo_orderkey         int     PRIMARY KEY,
        lo_linenumber       int,
        lo_custkey          int     FOREIGN KEY REFERENCES customer (c_custkey),
        lo_partkey          int     FOREIGN KEY REFERENCES part     (p_partkey),
        lo_suppkey          int     FOREIGN KEY REFERENCES supplier (s_suppkey),
        lo_orderdate        int     FOREIGN KEY REFERENCES date     (d_datekey),
        lo_orderpriority    string,
        lo_shippriority     string,
        lo_quantity         int,
        lo_extendedprice    int,
        lo_ordtotalprice    int,
        lo_discount         int,
        lo_revenue          int,
        lo_supplycost       int,
        lo_tax              int,
        lo_commitdate       int     FOREIGN KEY REFERENCES date     (d_datekey),
        lo_shipmode         string
    """
},
"tpch": {
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
},
"ch": {
    "supplier"  : r"""
        su_suppkey       int     PRIMARY KEY,
        su_name          string,
        su_address       string,
        su_nationkey     int     FOREIGN KEY REFERENCES nation  (n_nationkey),
        su_phone         string,
        su_acctbal       float,
        su_comment       string
    """,
    "nation"    : r"""
        n_nationkey     int     PRIMARY KEY,
        n_name          string,
        n_regionkey     int     FOREIGN KEY REFERENCES region  (r_regionkey),
        n_comment       string
    """,
    "region"    : r"""
        r_regionkey     int     PRIMARY KEY,
        r_name          string,
        r_comment       string
    """,
    "warehouse" : r"""
        w_id            int     PRIMARY KEY,
        w_name          string,
        w_street_1      string,
        w_street_2      string,
        w_city          string,
        w_state         string,
        w_zip           string,
        w_tax           float,
        w_ytd           float
    """,
    "district" : r"""
        d_id            int     PRIMARY KEY,
        d_w_id          int     FOREIGN KEY REFERENCES warehouse  (w_id),
        d_name          string,
        d_street_1      string,
        d_street_2      string,
        d_city          string,
        d_state         string,
        d_zip           string,
        d_tax           float,
        d_ytd           float,
        d_next_o_id     int64
    """,
    "stock" : r"""
        s_i_id           int     PRIMARY KEY     FOREIGN KEY REFERENCES item  (i_id),
        s_w_id          int     FOREIGN KEY REFERENCES warehouse  (w_id),
        s_quantity      int,
        s_dist_01       string,
        s_dist_02       string,
        s_dist_03       string,
        s_dist_04       string,
        s_dist_05       string,
        s_dist_06       string,
        s_dist_07       string,
        s_dist_08       string,
        s_dist_09       string,
        s_dist_10       string,
        s_ytd           float,
        s_order_cnt     int,
        s_remote_cnt    int,
        s_data          string,
        s_su_suppkey    int     FOREIGN KEY REFERENCES supplier  (su_suppkey)
    """,
    "item" : r"""
        i_id            int     PRIMARY KEY,
        i_im_id         int,
        i_name          string,
        i_price         float,
        i_data          string
    """,
    "orderline" : r"""
        ol_o_id         int64   PRIMARY KEY FOREIGN KEY REFERENCES order  (o_id),
        ol_d_id         int     FOREIGN KEY REFERENCES order  (o_d_id),
        ol_w_id         int     FOREIGN KEY REFERENCES order  (o_w_id),
        ol_number       int,
        ol_i_id         int     FOREIGN KEY REFERENCES stock  (s_i_id),
        ol_supply_w_id  int     FOREIGN KEY REFERENCES stock  (s_w_id),
        ol_delivery_d   datetime,
        ol_quantity     int,
        ol_amount       float,
        ol_dist_info    string
    """,
    "order" : r"""
        o_id            int64   PRIMARY KEY,
        o_d_id          int     FOREIGN KEY REFERENCES customer  (c_d_id),
        o_w_id          int     FOREIGN KEY REFERENCES customer  (c_w_id),
        o_c_id          int     FOREIGN KEY REFERENCES customer  (c_id),
        o_entry_d       datetime,
        o_carrier_id    int,
        o_ol_cnt        int,
        o_all_local     int
    """,
    "customer" : r"""
        c_id            int     PRIMARY KEY,
        c_d_id          int     FOREIGN KEY REFERENCES district  (d_id),
        c_w_id          int     FOREIGN KEY REFERENCES district  (d_w_id),
        c_first         string,
        c_middle        string,
        c_last          string,
        c_street_1      string,
        c_street_2      string,
        c_city          string,
        c_state         string,
        c_zip           string,
        c_phone         string,
        c_since         datetime,
        c_credit        string,
        c_credit_lim    float,
        c_discount      float,
        c_balance       float,
        c_ytd_payment   float,
        c_payment_cnt   int,
        c_delivery_cnt  int,
        c_data          string,
        c_n_nationkey   int     FOREIGN KEY REFERENCES nation  (n_nationkey)
    """,
    "neworder" : r"""
        no_o_id         int64   PRIMARY KEY     FOREIGN KEY REFERENCES order  (o_id),
        no_d_id         int     FOREIGN KEY REFERENCES order  (o_d_id),
        no_w_id         int     FOREIGN KEY REFERENCES order  (o_w_id)
    """,
    "history" : r"""
        h_c_id          int     PRIMARY KEY FOREIGN KEY REFERENCES customer  (c_id),
        h_c_d_id        int     FOREIGN KEY REFERENCES customer  (c_d_id),
        h_c_w_id        int     FOREIGN KEY REFERENCES customer  (c_w_id),
        h_d_id          int     FOREIGN KEY REFERENCES district  (d_id),
        h_w_id          int     FOREIGN KEY REFERENCES district  (d_w_id),
        h_date          datetime,
        h_amount        float,
        h_data          string
    """
},
"genom": {
    "broad": r"""
        filename        string,
        chrom           string,
        start           int64,
        stop            int64,
        name            string,
        score           int,
        strand          string,
        signal          float,
        pvalue          float,
        qvalue          float
    """,
        "broad2": r"""
        filename2       string,
        chrom2          string,
        start2          int64,
        stop2           int64,
        name2           string,
        score2          int,
        strand2         string,
        signal2         float,
        pvalue2         float,
        qvalue2         float
    """
},
"genom_32b": {
    "broad": r"""
        filename        string,
        chrom           string,
        start           int,
        stop            int,
        name            string,
        score           int,
        strand          string,
        signal          float,
        pvalue          float,
        qvalue          float
    """,
        "broad2": r"""
        filename2       string,
        chrom2          string,
        start2          int,
        stop2           int,
        name2           string,
        score2          int,
        strand2         string,
        signal2         float,
        pvalue2         float,
        qvalue2         float
    """
}
}

def run(cmd):
    print(cmd)
    if do_execute:
        process = subprocess.Popen(["time", "bash", "-c", cmd], stdout=subprocess.PIPE)
        output, error = process.communicate()
        if output:
            output = output.decode('ascii');
            print("output: " + output)
        if error:
            error = error.decode('ascii');
            print("error : " + error)
        return output


def clean_up(file):
    if do_clean:
        print("rm " + file + " # Deleting file: " + file)
        os.remove(file)
    else:
        print("# TODO: delete file: " + file)


def raw_filename(relName):
    return "raw/" + relName + ".tbl"

def prepare_attr(relName, attrName, attrIndex, type, relPathName, delim="'|'"):
    raw_file = raw_filename(relName)
    # extract column
    argNo = attrIndex + 1
    run(r"""# prepare """ + relName + "." + attrName)
    # if (attrName in ["lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey", "lo_suppkey", "lo_orderdate"]): return;
    columnFileTxt = attrName + ".csv"
    columnFileBin = relName + ".csv." + attrName
    # extract_cmd = "cat " + relName + ".tbl | cut -d " + delim + " -f" + str(attrIndex + 1) + " > " + columnFileTxt
    # run(extract_cmd)
    # convert to binary based on type
    pelago_type = type
    constraints = []
    if type == "int" or type == "boolean": # FIXME: do we handle bools as ints ? should we do something better?
        # for ints, just parse it
        # extract_cmd = "cat " + relName + ".tbl | cut -d " + delim + " -f" + str(attrIndex + 1) + " > " + columnFileTxt
        # run(extract_cmd)
        # parse_cmd = r"""perl -pe '$_=pack"l",$_' < """ + columnFileTxt + r""" > """ + columnFileBin
        parse_cmd = "cut -d " + delim + " -f" + str(argNo) + " " + raw_file + " | " + r"""perl -pe '$_=pack"l",$_' > """ + columnFileBin
        run(parse_cmd)
        pelago_type = "int" # FIXME: booleans should NOT be considered ints
    elif type == "int64": # FIXME: do we handle bools as ints ? should we do something better?
        parse_cmd = "cut -d " + delim + " -f" + str(argNo) + " " + raw_file + " | " + r"""perl -pe '$_=pack"q",$_' > """ + columnFileBin
        run(parse_cmd)
        pelago_type = "int64" # FIXME: booleans should NOT be considered ints
    elif type == "string":
        columnFileDict = columnFileBin + ".dict"
        build_dict_cmd = "cut -d " + delim + " -f" + str(argNo) + " " + raw_file + " | tee " + columnFileTxt + r""" | sort -S1G --parallel=24 -u | awk '{printf("%s:%d\n", $0, NR-1)}' | tee """ + columnFileDict + r""" | wc -l"""
        line_cnt = run(build_dict_cmd)
        first = run(r"""head -n 1 """ + columnFileDict)
        last  = run(r"""tail -n 1 """ + columnFileDict)
        constraints.append({
            "column": attrName,
            # Use rsplit as we only know that the last ':' is our delimeter
            "min": first.rsplit(":", 1)[0],
            "max": last.rsplit(":", 1)[0],
            "type": "range"
        })
        constraints.append({
            "columns": [attrName],
            "values": int(line_cnt),
            "type": "distinct_cnt"
        })
       	parse_cmd = r"""awk -F: 'FNR==NR {dict[$1]=$2; next} {$1=($1 in dict) ? dict[$1] : $1}1' """ + columnFileDict + r""" """ + columnFileTxt + r""" | perl -pe '$_=pack"l",$_' > """ + columnFileBin
        run(parse_cmd)
        clean_up(columnFileTxt)
        pelago_type = "dstring"
    elif type == "date":
        parse_cmd = "cut -d " + delim + " -f" + str(argNo) + " " + raw_file + " | " + r"""perl -MTime::Piece -pE '$_=pack("q",Time::Piece->strptime($_, "%Y-%m-%d\n")->epoch * 1000)' > """ + columnFileBin
        run(parse_cmd)
    elif type == "datetime":
        parse_cmd = "cut -d " + delim + " -f" + str(argNo) + " " + raw_file + " | " + r"""perl -MTime::Piece -pE '$_=pack("q",(eval{Time::Piece->strptime($_, "%Y-%m-%d %H:%M:%S\n")->epoch} or do { 0 } ) * 1000)' > """ + columnFileBin
        run(parse_cmd)
    elif type == "float":
        parse_cmd = "cut -d " + delim + " -f" + str(argNo) + " " + raw_file + " | " + r"""perl -pe '$_=pack"d",$_' > """ + columnFileBin
        run(parse_cmd)
    else:
        assert(False)  # Unknown type!
    return ({
        "type": {
            "type": pelago_type
        },
        "relName": relPathName,
        "attrName": attrName,
        "attrNo": argNo
    }, constraints)

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

def prepare_constraints_and_attribute(ind, c, relName, relPathName, schemaName, delim="'|'"):
    (att, con) = prepare_attr(relName, c[0], ind, c[1], relPathName, delim)
    if len(c) > 2 and len(c[2]) > 0:
        con.extend(create_constraint(c[0], c[2], lambda x: build_relname(schemaName, x)))
    return (att, con)

def count_lines(relName):
    with open(raw_filename(relName)) as f:
        return sum(1 for _ in f)

def prepare_rel(relName, namespace, relFolder, schemaName, delim="'|'"):
    schema = namespace[relName]
    columns = [attr.split(None, 2) for attr in schema.split(",")]
    relPathName = relFolder + "/" + schemaName + "/" + relName + ".csv"
    linehint = pool.apply_async(count_lines, (relName,));
    
    results = []
    attrs = []
    constraints = []
    for (ind, c) in enumerate(columns):
        results.append(pool.apply_async(prepare_constraints_and_attribute, (ind, c, relName, relPathName, schemaName, delim)))
        
    for r in results:
        tmp = r.get()
        attrs.append(tmp[0])
        constraints.extend(tmp[1])

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
                "linehint": linehint.get()
            },
            "constraints": constraints
        }
    }

parser = argparse.ArgumentParser(description=
r"""Convert flat text files to columnar binary files.

To use, navigate to the folder containing the *.tbl files, and execute: \n
python """ + __file__ + """ --schema-name ssbm1000 --ddl ssb
""")
parser.add_argument('--rel-folder', default="inputs")
parser.add_argument('--schema-name', required=True)
parser.add_argument('--ddl', choices=ddl.keys(), required=True)
parser.add_argument('--dry-run', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    relFolder  = args.rel_folder
    schemaName = args.schema_name
    
    namespace  = ddl[args.ddl]
    do_execute = not args.dry_run
    
    catalog    = {}
    # delim      = r"""$'\t'"""
    delim      = r"""'|'"""
    # print(json.dumps(prepare_rel("date", dates), indent = 4))
    # print(json.dumps(prepare_rel("customer", customer), indent = 4))
    # print(json.dumps(prepare_rel("supplier", supplier), indent = 4))
    # print(json.dumps(prepare_rel("part", part), indent = 4))
    # print(json.dumps(prepare_rel("lineorder", lineorder), indent = 4))
    
    res = []
    for name in namespace:
        res.append(pool.apply_async(prepare_rel, (name, namespace, relFolder, schemaName, delim)))
    
    for r in res:
        catalog.update(r.get())

    with open('catalog.json', 'w') as out:
        out.write(json.dumps(catalog, indent = 4))
        out.write('\n')

