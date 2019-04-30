import os
import glob
import errno
import sys
import datetime
import subprocess
import shlex

# Google
# from __future__ import print_function
import pickle
import os.path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

gsheet_creds = None
gsheet_service = None

expr_time = str(datetime.datetime.now())

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = "/home/raza/gcred.json"


aeoulus_exe = "./../../../opt/aeolus/aeolus-server"
DEBUG = False
GSHEET = True
FILE = True
STDOUT = False
result_dir = "./results"


# YCSB
YCSB_SPREADSHEET = '1OtICMs90trphneA7rFLWx3S3XVyzzp_Z9QGXuA8UqDs'
workers = [1, 2, 4, 8, 16, 18, 36, 54, 72]
# zipf Theta

# theta_range = [0, 1]
theta_range = [0.5, 0.5]
theta_step = 0.25

# Write Threshold
# write_thresh_range = [0, 1]
write_thresh_range = [0, 0.5]
write_thresh_step = 0.5
# Columns
num_cols_range = [4, 12]
num_cols_step = 4


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
def append_to_gsheet(values, spreadsheet_id=YCSB_SPREADSHEET, range="perf!A1:AB1", valueInputOption="USER_ENTERED"):
    body = {
        'values': [values]
    }
    result = gsheet_service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id, range='ycsb!A1:R1',
        valueInputOption="USER_ENTERED", body=body).execute()
    # print('{0} cells
    # appended.'.format(result.get('updates').get('updatedCells')))


def format_ycsb_gsheet(out, err):

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
    tps_st = stats.find("Socket-1: TPS") + len("Socket-1: TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    s1_tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # sock2
    tps_st = stats.find("Socket-2: TPS") + len("Socket-2: TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    s2_tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # sock3
    tps_st = stats.find("Socket-3: TPS") + len("Socket-3: TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    s3_tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # sock4
    tps_st = stats.find("Socket-4: TPS") + len("Socket-4: TPS") + 1
    tps_end = stats[tps_st:].find('\n')
    s4_tps = float(stats[tps_st: tps_st + tps_end].split(' ')[0])

    # DateTime   Commit ID   Commit Message  #Cores Zipf Theta  Write Threshold
    # Benchmark   Comments    Runtime (sec)   #Commits   #Aborts
    #   #Txns  Throughput (mTPS)
    # socket_1_tps    socket_2_tps    socket_3_tps    socket_4_tps
    return [cmt, abrt, txn, tps, s1_tps, s2_tps, s3_tps, s4_tps]


def frange(start, stop, step):
    x = start
    while x <= stop:
        yield x
        x += step


def test_ycsb(comments="", runtime=60):
    exp_counter = 0
    for col in frange(num_cols_range[0], num_cols_range[1], num_cols_step):
        if col == 0:
            col = 1
        for w in workers:
            exe = aeoulus_exe + " " + "-w " + \
                str(w) + " " + "-c " + str(col)
            # for r in xrange(int(write_thresh_range[0]*10),
            #                int((write_thresh_range[1]*10)+1), 2):
            # for r in write_thresh:
            for r in frange(write_thresh_range[0], write_thresh_range[1],
                            write_thresh_step):
                r_exe = exe + " -r " + str(r)
                for t in frange(theta_range[0], theta_range[1], theta_step):
                    t_exe = r_exe + " -t " + str(t)
                    print t_exe
                    if not DEBUG:
                        args = shlex.split(t_exe)
                        proc = subprocess.Popen(
                            args, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
                        output, err = proc.communicate()
                        if GSHEET:
                            val = [expr_time, "-", "-",
                                   str(w), str(t), str(
                                       r), "YCSB", col, comments,
                                   runtime]
                            gsheet_init2()
                            val += format_ycsb_gsheet(output, err)
                            append_to_gsheet(
                                val, spreadsheet_id=YCSB_SPREADSHEET)
                        if STDOUT:
                            print "---------------stdout------------"
                            print output
                            if err and len(err) > 0:
                                print "---------------stderr------------"
                                print err
                        if FILE:
                            out_file_path = os.path.join(
                                result_dir, "ycsb_" + str(col) + "_" + str(w) + "_" + str(r) + "_" + str(t))
                            out_file = open(out_file_path, 'w')
                            out_file.write("---------------stdout------------")
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
    print format_ycsb_gsheet(inn.read(), "")


if __name__ == "__main__":
    gsheet_init2()
    test_ycsb(comments="delta switch off", runtime=60)
