import os
import glob
import errno
import sys
import datetime
import subprocess
import shlex
import time
import multiprocessing

# Google
# from __future__ import print_function
import pickle
import os.path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import platform

gsheet_creds = None
gsheet_service = None

expr_time = str(datetime.datetime.now())
expr_time_epoch = str(int(time.time()))

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = "/home/raza/gcred.json"


aeoulus_exe = "./../../../../opt/aeolus/aeolus-server"
DEBUG = False
GSHEET = True
FILE = False
STDOUT = True
result_dir = "./results"

repeat_expr_times = 5

AEOLUS_SPREADSHEET = '1OtICMs90trphneA7rFLWx3S3XVyzzp_Z9QGXuA8UqDs'

ycsb_range = 'ycsb_new!A1:T1'
tpcc_range = 'tpcc_new!A1:O1'


# General
workers = [1, 2, 4, 8, 16, 18, 32, 36, 54, 64, 72]
num_partitions = [1, 2, 3, 4]
layout_column_store = ["true", "false"]

# YCSB
# theta_range = [0, 1]
theta_range = [0.5, 0.5]
theta_step = 0.25

# Write Threshold
# write_thresh_range = [0, 1]
write_thresh_range = [0.0, 1.0]
write_thresh_step = 0.25
# Columns
num_cols = [1, 2, 4, 8, 16]
#num_cols_range = [0, 12]
#num_cols_step = 4
# --------------

# TPCC


def gsheet_init2(default_token_file="/home/raza/gtoken.pickle"):
    global gsheet_service, gsheet_creds
    gsheet_creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gsheet_service = build('sheets', 'v4', credentials=gsheet_creds)


def gsheet_init(default_token_file="/home/raza/gtoken.pickle"):
    global gsheet_service, gsheet_creds
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(default_token_file):
        with open(default_token_file, 'rb') as token:
            gsheet_creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not gsheet_creds or not gsheet_creds.valid:
        if gsheet_creds and gsheet_creds.expired and gsheet_creds.refresh_token:
            gsheet_creds.refresh(Request())
        else:
            # flow = InstalledAppFlow.from_client_secrets_file(
            #    SERVICE_ACCOUNT_FILE, SCOPES)
            # gsheet_creds = flow.run_local_server()
            gsheet_creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        # Save the credentials for the next run
        with open(default_token_file, 'wb') as token:
            pickle.dump(gsheet_creds, token)

    gsheet_service = build('sheets', 'v4', credentials=gsheet_creds)


# expects values formatted according to the sheet
def append_to_gsheet(values, spreadsheet_id=AEOLUS_SPREADSHEET, range="perf!A1:AB1", valueInputOption="USER_ENTERED"):
    body = {
        'values': [values]
    }
    result = gsheet_service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id, range=range,
        valueInputOption="USER_ENTERED", body=body).execute()
    # print('{0} cells
    # appended.'.format(result.get('updates').get('updatedCells')))


def format_metrics_gsheet(out, err):

    # Gloal Stats
    glbl_st = out.find("---- GLOBAL ----") + len("---- GLOBAL ----")
    glbl_end = out.find("------------ END WORKER STATS ------------")
    stats = out[glbl_st + 1:glbl_end]
    # num_txns
    txn_st = stats.find("# of txns") + len("# of txns") + 1
    txn_end = stats[txn_st:].find('\n')
    txn = float(stats[txn_st: txn_st + txn_end].split(' ')[0])

    # num_cmt
    cmt_st = stats.find("# of commits") + len("# of commits") + 1
    cmt_end = stats[cmt_st:].find('\n')
    cmt = float(stats[cmt_st: cmt_st + cmt_end].split(' ')[0])

    # num_abrt
    abrt_st = stats.find("# of aborts") + len("# of aborts") + 1
    abrt_end = stats[abrt_st:].find('\n')
    abrt = float(stats[abrt_st: abrt_st + abrt_end].split(' ')[0])

    # tps
    tps_st = stats.find("TPS") + len("TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # Socket level
    sock_st = out.find("---- SOCKET ----") + len("---- SOCKET ----")
    sock_end = out.find("---- GLOBAL ----")
    stats = out[sock_st + 1:sock_end]

    # sock1
    tps_st = stats.find("Socket-0: TPS") + len("Socket-1: TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    s1_tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # sock2
    tps_st = stats.find("Socket-1: TPS") + len("Socket-2: TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    s2_tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # sock3
    tps_st = stats.find("Socket-2: TPS") + len("Socket-3: TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    s3_tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # sock4
    tps_st = stats.find("Socket-3: TPS") + len("Socket-4: TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    s4_tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # # Delta 0
    # delta_0_st = out.find("[DeltaStore # 0]") + len("[DeltaStore # 0]")
    # delta_0_end = out.find("[DeltaStore # 1]")
    # delta_0_stats = out[delta_0_st + 1:delta_0_end]

    # # [DeltaStore # 0] Number of GC Requests: 13361
    # # [DeltaStore # 0] Number of Successful GC Resets: 13360
    # # [DeltaStore # 0] Number of Operations: 1595390

    # d0_req_st = delta_0_stats.find(
    #     "Number of GC Requests:") + len("Number of GC Requests:") + 1
    # d0_req_end = delta_0_stats[d0_req_st:].find('\n')
    # d0_req = int(
    #     delta_0_stats[d0_req_st: d0_req_st + d0_req_end].split(' ')[0])

    # d0_suc_st = delta_0_stats.find(
    #     "Number of Successful GC Resets:") + len("Number of Successful GC Resets:") + 1
    # d0_suc_end = delta_0_stats[d0_suc_st:].find('\n')
    # d0_suc = int(
    #     delta_0_stats[d0_suc_st: d0_suc_st + d0_suc_end].split(' ')[0])

    # d0_op_st = delta_0_stats.find(
    #     "Number of Operations:") + len("Number of Operations:") + 1
    # d0_op_end = delta_0_stats[d0_op_st:].find('\n')
    # d0_op = int(delta_0_stats[d0_op_st: d0_op_st + d0_op_end].split(' ')[0])

    # # Delta 1
    # delta_1_st = out.find("[DeltaStore # 1]") + len("[DeltaStore # 1]")
    # out_inter = out[delta_1_st + 1:]
    # delta_1_end = out_inter.find("Memory Consumption:")
    # delta_1_stats = out_inter[:delta_1_end]

    # d1_req_st = delta_1_stats.find(
    #     "Number of GC Requests:") + len("Number of GC Requests:") + 1
    # d1_req_end = delta_1_stats[d1_req_st:].find('\n')
    # d1_req = int(
    #     delta_1_stats[d1_req_st: d1_req_st + d1_req_end].split(' ')[0])

    # d1_suc_st = delta_1_stats.find(
    #     "Number of Successful GC Resets:") + len("Number of Successful GC Resets:") + 1
    # d1_suc_end = delta_1_stats[d1_suc_st:].find('\n')
    # d1_suc = int(
    #     delta_1_stats[d1_suc_st: d1_suc_st + d1_suc_end].split(' ')[0])

    # d1_op_st = delta_1_stats.find(
    #     "Number of Operations:") + len("Number of Operations:") + 1
    # d1_op_end = delta_1_stats[d1_op_st:].find('\n')
    # d1_op = int(delta_1_stats[d1_op_st: d1_op_st + d1_op_end].split(' ')[0])

    # DateTime   Commit ID   Commit Message  #Cores Zipf Theta  Write Threshold
    # Benchmark   Comments    Runtime (sec)   #Commits   #Aborts
    #   #Txns  Throughput (mTPS)
    # socket_1_tps    socket_2_tps    socket_3_tps    socket_4_tps
    # , d0_req, d0_suc, d0_op, d1_req, d1_suc, d1_op]
    return [cmt, abrt, txn, tps, s1_tps, s2_tps, s3_tps, s4_tps]


def frange(start, stop, step):
    x = start
    while x <= stop:
        yield x
        x += step


def test_ycsb(layout_col="true", num_partitions=1, num_workers=1, comments="", runtime=60):
    exp_counter = 0
    for num_exp in xrange(0, repeat_expr_times):
        for col in num_cols:
            exe = aeoulus_exe + " --benchmark=0"
            exe += " --layout_column_store=" + layout_col
            exe += " --delta-size=4"
            exe += " --runtime=" + str(runtime)
            exe += " --num-partitions=" + str(num_partitions)
            exe += " --num-workers=" + str(num_workers)

            exe += " --ycsb_num_cols=" + str(col)
            for r in frange(write_thresh_range[0], write_thresh_range[1],
                            write_thresh_step):
                r_exe = exe + " --ycsb_write_ratio=" + str(r)
                for t in frange(theta_range[0], theta_range[1], theta_step):
                    t_exe = r_exe + " --ycsb_zipf_theta=" + str(t)

                    comment_w_exe = comments + " \n"
                    print num_exp, "--", t_exe
                    if not DEBUG:
                        args = shlex.split(t_exe)
                        proc = subprocess.Popen(
                            args, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
                        output, err = proc.communicate()
                        if GSHEET:
                            layout_str = "COLUMNAR"
                            if(layout_col != "true"):
                                layout_str = "ROW"
                            val = [expr_time, platform.uname()[1], "-", "-",
                                   str(num_workers), layout_str, str(t), str(
                                       r), "YCSB", col, "SUCCESS", comment_w_exe,
                                   runtime]
                            if len(err) > 10:
                                val[10] = "FAILED"
                                val[11] += "\n " + str(err)
                            else:
                                val += format_metrics_gsheet(output, err)
                                val += [exe]
                            gsheet_init2()
                            append_to_gsheet(
                                val, spreadsheet_id=AEOLUS_SPREADSHEET, range=ycsb_range)
                        if STDOUT:
                            print "---------------stdout------------"
                            print output
                            if err and len(err) > 0:
                                print "---------------stderr------------"
                                print err
                        if FILE:
                            out_file_path = os.path.join(
                                result_dir, expr_time_epoch + "_ycsb_" + str(col) + "_" +
                                str(w) + "_" + str(r) + "_" + str(t) + "_"
                                + str(num_exp))
                            out_file = open(out_file_path, 'w')
                            out_file.write(
                                "---------------stdout------------")
                            out_file.write(output)
                            if err and len(err) > 0:
                                out_file.write(
                                    "---------------stderr------------")
                                out_file.write(err)
                            out_file.close()
                    exp_counter += 1
    print "Total Experiements completed:", exp_counter


def test_tpcc(layout_col="true", num_partitions=1, num_workers=1, comments="", runtime=60):
    exp_counter = 0
    for num_exp in xrange(0, repeat_expr_times):
        exe = aeoulus_exe + " --benchmark=1"
        exe += "  --tpcc_num_wh=" + str(multiprocessing.cpu_count())
        exe += " --layout_column_store=" + layout_col
        exe += " --delta-size=4"
        exe += " --runtime=" + str(runtime)
        exe += " --num-partitions=" + str(num_partitions)
        exe += " --num-workers=" + str(num_workers)
        comment_w_exe = comments + " \n"
        print num_exp, "--", exe
        if not DEBUG:
            args = shlex.split(exe)
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            output, err = proc.communicate()
            if GSHEET:
                layout_str = "COLUMNAR"
                if(layout_col != "true"):
                    layout_str = "ROW"
                val = [expr_time, platform.uname()[1], "-", "-",
                       str(num_workers), layout_str, "TPCC", "SUCCESS", comment_w_exe, runtime]
                gsheet_init2()
                if len(err) > 10:
                    val[7] = "FAILED"
                    val[8] += "\n " + str(err)
                else:
                    val += format_metrics_gsheet(output, err)
                    val += [exe]
                append_to_gsheet(
                    val, spreadsheet_id=AEOLUS_SPREADSHEET,
                    range=tpcc_range)
            if STDOUT:
                print "---------------stdout------------"
                print output
                if err and len(err) > 0:
                    print "---------------stderr------------"
                    print err
            if FILE:
                out_file_path = os.path.join(
                    result_dir, expr_time_epoch + "_tpcc_" + str(w) + "_" + str(num_exp))
                out_file = open(out_file_path, 'w')
                out_file.write(
                    "---------------stdout------------")
                out_file.write(output)
                if err and len(err) > 0:
                    out_file.write(
                        "---------------stderr------------")
                    out_file.write(err)
                out_file.close()
        exp_counter += 1
    print "Total Experiements completed:", exp_counter


def test():
    path = os.path.join("./results.o", "ycsb_1_1_0_5")
    inn = open(path, 'r')
    print format_metrics_gsheet(inn.read(), "")


def test_all_tpcc():
    num_p = 2
    if(platform.uname()[1] == "diascld33"):
        num_p = 4
    print "Setting # of Partitions to", num_p

    for layout in layout_column_store:
        for wrkr in workers:
            test_tpcc(layout_col=layout, num_partitions=num_p, num_workers=wrkr,
                      comments="neworder txn", runtime=30)


def test_all_ycsb():
    num_p = 2
    if(platform.uname()[1] == "diascld33"):
        num_p = 4
    print "Setting # of Partitions to", num_p
    for layout in layout_column_store:
        # for n_part in num_partitions:
        for wrkr in workers:
            test_ycsb(layout_col=layout, num_partitions=num_p, num_workers=wrkr,
                      comments="partition_local_zipf", runtime=30)


if __name__ == "__main__":
    gsheet_init2()
    test_all_tpcc()
    # test_all_ycsb()
