import subprocess
import shlex
import os

# zipf, worker, write_thresh

result_dir = "./tpcc_results"
aeoulus_exe = "./../../../opt/aeolus/aeolus-server -b 1"

worker_range = [1, 72]


# workers = [1, 2, 4, 8, 16, 18, 36, 54, 72]
workers = [1, 2, 4, 8, 16, 18, 36, 54, 72]

debug = False


def main():
    # for w in xrange(worker_range[0], worker_range[1]+1, 18):
    for w in workers:
        exe = aeoulus_exe + " " + "-w " + str(w)
        # for r in xrange(int(write_thresh_range[0]*10),
        #                int((write_thresh_range[1]*10)+1), 2):

        print exe
        if not debug:
            out_file_path = os.path.join(result_dir, "tpcc_" + "_" + str(w))
            out_file = open(out_file_path, 'w')
            args = shlex.split(exe)
            subprocess.call(args, stdout=out_file)


if __name__ == "__main__":
    main()
