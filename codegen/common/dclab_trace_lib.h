// dclab_trace_lib.h 
//
// This is a simple interface for user-mode code to control kernel/user tracing and add markers
//

#ifndef __DCLAB_TRACE_LIB_H__
#define __DCLAB_TRACE_LIB_H__

// This is the definitive list of raw trace 12-bit event numbers
#define NopNum           0x000
#define RdtscNum         0x001
#define GetTODNum        0x002
#define VarlenLoNum      0x010
#define VarlenHiNum      0x1FF

// Variable-length starting numbers. 
// Middle hex digit will become 1..8
#define FileNameNum      0x001
#define PidNameNum       0x002
#define MethodNameNum    0x003
#define TrapNameNum      0x004
#define InterruptNameNum 0x005
#define Syscall64NameNum 0x008
#define Syscall32NameNum 0x00C
#define PacketsNum       0x100

// Specials are point events
#define UserPidNum       0x200
#define RpcidReqNum      0x201
#define RpcidRespNum     0x202
#define RpcidMidNum      0x203
#define RpcidRxPktNum    0x204
#define RpcidTxPktNum    0x205
#define RunnableNum      0x206
#define IPINum           0x207
#define MwaitNum         0x208
#define LockWaitNum      0x209
#define Mark_a           0x20A
#define Mark_b           0x20B
#define Mark_c           0x20C
#define Mark_d           0x20D
  // available           0x20E
  // available           0x20F

// These are in blocks of 256 or 512 numbers
#define TrapNum          0x400
#define InterruptNum     0x500
#define TrapRetNum       0x600
#define InterruptRetNum  0x700
#define Syscall64Num     0x800
#define Syscall64RetNum  0xA00
#define Syscall32um      0xC00
#define Syscall32RetNum  0xE00

namespace dclab_trace {
  bool test();
  void go(const char* process_name);
  void goipc(const char* process_name);
  void stop(const char* fname);
  void mark_a(const char* label);
  void mark_b(const char* label);
  void mark_c(const char* label);
  void mark_d(unsigned long int n);
  void addevent(unsigned long int eventnum, unsigned long int arg);
}

#endif	// __DCLAB_TRACE_LIB_H__


