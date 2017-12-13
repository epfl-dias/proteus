// Little user-mode library program to control dclab_tracing 
// dick sites 2017.08.25
// 2017.11.16 dsites Updated to include instructions per cycle IPC flag
//

#include <stdio.h>
#include <stdlib.h>     // exit
#include <string.h>
#include <time.h>	// nanosleep
#include <unistd.h>     // getpid gethostname
#include <x86intrin.h>
#include <sys/time.h>   // gettimeofday
#include <sys/types.h>	 

#include "dclab_trace_lib.h"

#include "basetypes.h"
#include "dclab_control_names.h"


#define TRACE_OFF 0
#define TRACE_ON 1
#define TRACE_FLUSH 2
#define TRACE_RESET 3
#define TRACE_STAT 4
#define TRACE_GETCOUNT 5
#define TRACE_GETWORD 6
#define TRACE_INSERT1 7
#define TRACE_INSERTN 8
#define TRACE_GETIPCWORD 9


namespace {

typedef uint64 u64;
#define IPC_Flag 0x80

// Module must be at least this version number for us to run
static const u64 kMinModuleVersionNumber = 2;

// Number of u64 values per trace block
static const int kTraceBufSize = 8192;

// Number of u64 values per IPC block, one u8 per u64 in trace buf
static const int kIpcBufSize = kTraceBufSize >> 3;

// Globals for mapping cycles to gettimeofday
int64 start_cycles = 0;
int64 stop_cycles = 0;
int64 start_usec = 0;
int64 stop_usec = 0;

// Useful utility routines
int64 GetUsec() {struct timeval tv; gettimeofday(&tv, NULL);
                 return (tv.tv_sec * 1000000l) + tv.tv_usec;}

// Read the cycle counter and gettimeofday() close together, returning both
void GetTimePair(int64* usec, int64* cycles) {
  uint64 startcy, stopcy;
  int64 gtodusec, elapsedcy;
  // Do more than once if we get an interrupt or other big delay in the middle of the loop
  do {
    startcy = __rdtsc();
    gtodusec = GetUsec();
    stopcy = __rdtsc();
    elapsedcy = stopcy - startcy;
    // In a quick test on an Intel i3 chip, GetUsec() took about 150 cycles
    // printf("%ld elapsed cycles\n", elapsedcy);
  } while (elapsedcy > 20000);  // About 8 usec at 2.5GHz
  *usec = gtodusec;
  *cycles = startcy;
}


// For the trace_control system call,
// arg is declared to be u64. In reality, it is either a u64 or
// a pointer to a u64, depending on the command. Caller casts as
// needed, and the command implementations in dclab_trace_mod
// cast back as needed.

// u64 (*dclab_trace_control)(u64 command, u64 arg); 
//#define __NR_dclab_control 511
//static inline _syscall2(u64, dclab_control, u64, command, u64, arg);

#define __NR_dclab_control 511
inline u64 DoControl(u64 command, u64 arg)
{
    u64 retval;
    asm volatile
    (
        "syscall"
        : "=a" (retval)
        : "0"(__NR_dclab_control), "D"(command), "S"(arg)
        : "cc", "rcx", "r11", "memory"
    );
    return retval;
}

// Sleep for n milliseconds
void msleep(int msec) {
  struct timespec ts;
  ts.tv_sec = msec / 1000;
  ts.tv_nsec = (msec % 1000) * 1000000;
  nanosleep(&ts, NULL);
}

// Single static buffer. In real production code, this would 
// all be std::string value, or something else at least as safe.
static const int kMaxDateTimeBuffer = 32;
static char gTempDateTimeBuffer[kMaxDateTimeBuffer];

// Turn seconds since the epoch into yyyymmdd_hhmmss
// Not valid after January 19, 2038
const char* FormatSecondsDateTime(int32 sec) {
  // if (sec == 0) {return "unknown";}  // Longer spelling: caller expecting date
  time_t tt = sec;
  struct tm* t = localtime(&tt);
  sprintf(gTempDateTimeBuffer, "%04d%02d%02d_%02d%02d%02d",
         t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
         t->tm_hour, t->tm_min, t->tm_sec);
  return gTempDateTimeBuffer;
}

// In a production environment, use std::string or something safer
static char tempLogFileName[256];

// Construct a name for opening a trace file, using name of program from command line
//   name: program_time_host_pid
const char* MakeTraceFileName(const char* argv0) {
  time_t tt;
  const char* timestr;
  char hostnamestr[256];
  int pid;

  const char* slash = strrchr(argv0, '/');
  // Point to first char of image name
  if (slash == NULL) {
    slash = argv0;
  } else {
    slash = slash + 1;  // over the slash
  }

  tt = time(NULL);
  timestr = FormatSecondsDateTime(tt);
  gethostname(hostnamestr, 256) ;
  hostnamestr[255] = '\0';
  pid = getpid();

  sprintf(tempLogFileName, "%s_%s_%s_%d.trace",
          slash, timestr, hostnamestr, pid);
  return tempLogFileName; 
}           

// Add a list of names to the trace
void EmitNames(const NumNamePair* ipair, u64 n) {
  u64 temp[9];		// One extra word for strcpy(56 bytes + '\n')
  const NumNamePair* pair = ipair;
  while (pair->name != NULL) {
    u64 bytelen = strlen(pair->name);
    if (bytelen > 56) {continue;}	// Error if too long. Drop
    u64 wordlen = 1 + ((bytelen + 7) / 8);
    // Build the initial word
    u64 n_with_length = n + (wordlen * 16);
    //         T             N                       ARG
    temp[0] = (0l << 44) | (n_with_length << 32) | (pair->number);
    memset(&temp[1], 0, 8 * sizeof(u64));
    strcpy((char*)&temp[1], pair->name);
    DoControl(TRACE_INSERTN, (u64)&temp[0]);
    ++pair;
  }
}

// Return false if the module is not loaded. No delay. No side effect on time.
bool TestModule() {
  // If module is not loaded, syscall 511 returns -ENOSYS (= -38)
  u64 retval = DoControl(TRACE_OFF, 0);
  if ((int64)retval < 0) {
    // Module is not loaded
    fprintf(stderr, "Module dclab_trace_mod.ko not loaded\n");
    return false;
  }
  if (retval < kMinModuleVersionNumber) {
    // Module is loaded but older version
    fprintf(stderr, "Module dclab_trace_mod.ko is version %ld. Need at least %ld\n",
      retval, kMinModuleVersionNumber);
    return false;
  }
  return true;
}

// Turn off tracing
// Complain and return false if module is not loaded
bool DoOff() {
  u64 retval = DoControl(TRACE_OFF, 0);
  msleep(20);	/* Wait 20 msec for any pending tracing to finish */
  if (retval != 0) {
    // Module is not loaded
    fprintf(stderr, "Module dclab_trace_mod.ko not loaded\n");
    return false;
  }
  // Get stop time pair with tracing off
  if (stop_usec == 0) {GetTimePair(&stop_usec, &stop_cycles);}
  // fprintf(stdout, "GetTimePair %ld %ld\n", stop_usec, stop_cycles);
  return true;
}

// Turn on tracing
// Complain and return false if module is not loaded
bool DoOn() {
  // Get start time pair with tracing off
  if (start_usec == 0) {GetTimePair(&start_usec, &start_cycles);}
  // fprintf(stdout, "GetTimePair %ld %ld\n", start_usec, start_cycles);
  u64 retval = DoControl(TRACE_ON, 0);
  if (retval != 1) {
    // Module is not loaded
    fprintf(stderr, "Module dclab_trace_mod.ko not loaded\n");
    return false;
  }
  return true;
}


// Initialize trace buffer with syscall/irq/trap names
// Module must be loaded. Tracing must be off
void DoInit(const char* process_name) {
  if (!TestModule()) {return;}		// No module loaded
  EmitNames(TrapNames, TrapNameNum);
  EmitNames(IrqNames, InterruptNameNum);
  EmitNames(Syscall64Names, Syscall64NameNum);

  // We want to force emit of the current pid name here
  int pid = getpid() & 0x0000ffff;
  u64 temp[3];
  u64 n_with_length = PidNameNum + (3 << 4);
  //         T             N                       ARG
  temp[0] = (0ll << 44) | (n_with_length << 32) | (pid);
  temp[1] = 0;
  temp[2] = 0;
  if (strlen(process_name) < 16) {
    strcpy((char*)&temp[1], process_name);
  } else {
    memcpy((char*)&temp[1], process_name, 16);
  }
  DoControl(TRACE_INSERTN, (u64)&temp[0]);

  // And then establish that pid on this CPU
  n_with_length = UserPidNum;
  //         T             N                       ARG
  temp[0] = (0l << 44) | (n_with_length << 32) | (pid);
  DoControl(TRACE_INSERT1, temp[0]);
}

// With tracing off, zero out the rest of each partly-used traceblock
// Module must be loaded. Tracing must be off
void DoFlush() {
  if (!TestModule()) {return;}		// No module loaded
  DoControl(TRACE_FLUSH, 0);
}

// Set up for a new tracing run
// Module must be loaded. Tracing must be off
void DoReset(u64 doing_ipc) {
  if (!TestModule()) {return;}		// No module loaded
  DoControl(TRACE_RESET, doing_ipc);
  start_usec = 0;
  stop_usec = 0;
  start_cycles = 0;
  stop_cycles = 0;
}


// Show some sort of tracing status
// Module must be loaded. Tracing may well be on
void DoStat() {
  u64 retval = DoControl(TRACE_STAT, 0);
  if (retval < 160) {
    fprintf(stderr, "Stat: %ld trace blocks used (%ldKB)\n", retval, retval << 6);
  } else {
    fprintf(stderr, "Stat: %ld trace blocks used (%ldMB)\n", retval, retval >> 4);
  }
}


// Dump the trace buffer to filename
// Module must be loaded. Tracing must be off
void DoDump(const char* fname) {
  if (!TestModule()) {return;}		// No module loaded
  DoControl(TRACE_FLUSH, 0);

  // Calculate mapping from cycles to usec. Anding is because cycles are
  // stored with cpu# in the high byte of traceblock[0]
  int64 start_cycles56 = start_cycles & 0x00fffffffffffffl;
  int64 stop_cycles56 =  stop_cycles & 0x00ffffffffffffffl;
  double m = (stop_usec - start_usec) * 1.0;
  if ((stop_cycles56 - start_cycles56) > 0.0) {m = m / (stop_cycles56 - start_cycles56);}
  // We expect m to be on the order of 1/3000

  FILE* f = fopen(fname, "wb");
  if (f == NULL) {
    fprintf(stderr, "%s did not open\n", fname);
    return;
  }

  u64 traceblock[kTraceBufSize];
  u64 ipcblock[kIpcBufSize];
  // get number of trace blocks
  u64 wordcount = DoControl(TRACE_GETCOUNT, 0);
  u64 blockcount = wordcount >> 13;

  // Loop on trace blocks
  for (int i = 0; i < blockcount; ++i) {
    u64 k = i * kTraceBufSize;  // Trace Word number to fetch next
    u64 k2 = i * kIpcBufSize;  	// IPC Word number to fetch next
    
    // Extract 64KB trace block
    for (int j = 0; j < kTraceBufSize; ++j) {
      traceblock[j] = DoControl(TRACE_GETWORD, k++);
    }

    // traceblock[0] already has cycle counter and cpu#

    // traceblock[1] already has flags in top byte
    uint8 flags = traceblock[1] >> 56;
    bool this_block_has_ipc = ((flags & IPC_Flag) != 0);

    // Set the interpolated gettimeofday time 
    int64 mid_usec = (traceblock[0] & 0x00ffffffffffffffl) - start_cycles56;
    mid_usec *= m;
    traceblock[1] |= mid_usec + start_usec;

    // For very first block, insert value of m as a double, for dump program to use
    // and clear traceblock[3], reserved for future use
    if (i == 0) {
      traceblock[2] = *(u64*)&m;
      traceblock[3] = 0;
    }
    fwrite(traceblock, 1, sizeof(traceblock), f);

    // For each 64KB traceblock that has IPC_Flag set, also read the IPC bytes
    if (this_block_has_ipc) {
      // Extract 8KB IPC block
      for (int j = 0; j < kIpcBufSize; ++j) {
        ipcblock[j] = DoControl(TRACE_GETIPCWORD, k2++);
      }
      fwrite(ipcblock, 1, sizeof(ipcblock), f);
    }
  }
  fclose(f);

  // Go ahead and set up for another trace
  DoControl(TRACE_RESET, 0);
}


// Exit this program
// Tracing must be off
void DoQuit() {
  DoOff();
  exit(0);
}

// Create a Mark entry
void DoMark(u64 n, u64 arg) {
  //         T             N                       ARG
  u64 temp = (0l << 44) | (n << 32) | (arg & 0x00000000FFFFFFFFl);
  DoControl(TRACE_INSERT1, temp);
}

// Create a Mark entry
void DoEvent(u64 eventnum, u64 arg) {
  //         T             N                       ARG
  u64 temp = ((eventnum & 0xFFF) << 32) | (arg & 0x00000000FFFFFFFFl);
  DoControl(TRACE_INSERT1, temp);
}

// Uppercase mapped to lowercase
// All unexpected characters mapped to '-'
//   - = 0x2D . = 0x2E / = 0x2F
// Base40 characters are _abcdefghijklmnopqrstuvwxyz0123456789-./
//                       0         1         2         3
//                       0123456789012345678901234567890123456789
// where the first is NUL.
static const char kToBase40[256] = {
   0,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,37,38,39, 
  27,28,29,30, 31,32,33,34, 35,36,38,38, 38,38,38,38, 

  38, 1, 2, 3,  4, 5, 6, 7,  8, 9,10,11, 12,13,14,15,
  16,17,18,19, 20,21,22,23, 24,25,26,38, 38,38,38,38, 
  38, 1, 2, 3,  4, 5, 6, 7,  8, 9,10,11, 12,13,14,15,
  16,17,18,19, 20,21,22,23, 24,25,26,38, 38,38,38,38, 

  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 

  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
  38,38,38,38, 38,38,38,38, 38,38,38,38, 38,38,38,38, 
};

static const char kFromBase40[40] = {
  '\0','a','b','c', 'd','e','f','g',  'h','i','j','k',  'l','m','n','o',
  'p','q','r','s',  't','u','v','w',  'x','y','z','0',  '1','2','3','4',
  '5','6','7','8',  '9','-','.','/', 
};

// Unpack six characters from 32 bits.
// str must be 8 bytes. We somewhat-arbitrarily capitalize the first letter
char* Base40ToChar(u64 base40, char* str) {
  base40 &= 0x00000000fffffffflu;	// Just low 32 bits
  memset(str, 0, 8);
  bool first_letter = true;
  // First character went in last, comes out first
  int i = 0;
  while (base40 > 0) {
    u64 n40 = base40 % 40;
    str[i] = kFromBase40[n40];
    base40 /= 40;
    if (first_letter && (1 <= n40) && (n40 <= 26)) {
      str[i] &= ~0x20; 		// Uppercase it
      first_letter = false;
    }
    ++i;
  }
  return str;
}

// Pack six characters into 32 bits. Only use a-zA-Z0-9.-/
u64 CharToBase40(const char* str) {
  int len = strlen(str);
  // If longer than 6 characters, take only the first 6
  if (len > 6) {len = 6;}
  u64 base40 = 0;
  // First character goes in last, comes out first
  for (int i = len - 1; i >= 0; -- i) {
    base40 = (base40 * 40) + kToBase40[str[i]];
  }
  return base40;
}

}  // End anonymous namespace

bool dclab_trace::test() {return TestModule();}
void dclab_trace::go(const char* process_name) {DoReset(0); DoInit(process_name); DoOn();}
void dclab_trace::goipc(const char* process_name) {DoReset(1); DoInit(process_name); DoOn();}
void dclab_trace::stop(const char* fname) {DoOff(); DoFlush(); DoDump(fname); DoQuit();}
void dclab_trace::mark_a(const char* label) {DoMark(Mark_a, CharToBase40(label));}
void dclab_trace::mark_b(const char* label) {DoMark(Mark_b, CharToBase40(label));}
void dclab_trace::mark_c(const char* label) {DoMark(Mark_c, CharToBase40(label));}
void dclab_trace::mark_d(uint64 n) {DoMark(Mark_d, n);}
void dclab_trace::addevent(uint64 eventnum, uint64 arg) {DoEvent(eventnum, arg);}

 

