import subprocess
import shlex
import os

# zipf, worker, write_thresh

result_dir = "./results"
aeoulus_exe = "./../../../opt/aeolus/aeolus-server"

worker_range = [1, 72]
theta_range = [0.5, 0.5]
write_thresh_range = [0.0, 0.5]
num_cols_range = [1, 11]


# workers = [1, 2, 4, 8, 16, 18, 36, 54, 72]
workers = [1, 2, 4, 8, 16, 18, 36, 54, 72]
write_thresh = [0, 1.0, 2.5, 5, 7.5, 10]

debug = False


def main():
    # for w in xrange(worker_range[0], worker_range[1]+1, 18):
    for cc in xrange(num_cols_range[0], num_cols_range[1]):
        for w in workers:
            exe = aeoulus_exe + " " + "-w " + str(w) + " " + "-c " + str(cc)
            # for r in xrange(int(write_thresh_range[0]*10),
            #                int((write_thresh_range[1]*10)+1), 2):
            for r in write_thresh:
                r_i = 0.0
                if(r != 0):
                    r_i = float(r) / 10
                r_exe = exe + " -r " + str(r_i)
                for t in xrange(int(theta_range[0]*10),
                                int((theta_range[1]*10)+1), 2):
                    t_i = 0.0
                    if(t != 0):
                        t_i = float(t) / 10
                    t_exe = r_exe + " -t " + str(t_i)
                    print t_exe
                    if not debug:
                        out_file_path = os.path.join(result_dir, "ycsb_"+str(cc)+"_"+str(w)+"_"+str(r)+"_"+str(t))
                        out_file = open(out_file_path, 'w')
                        args = shlex.split(t_exe)
                        subprocess.call(args, stdout=out_file)


if __name__ == "__main__":
    main()
