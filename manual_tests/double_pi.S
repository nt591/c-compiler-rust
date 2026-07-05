  .globl _get_pi
  .text
_get_pi:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $0, %rsp
  movsd  _pi(%rip), %xmm0
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .globl _main
  .text
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  call   _get_pi
  movsd  %xmm0, -8(%rbp)
  movsd  -8(%rbp), %xmm14
  movsd  %xmm14, -16(%rbp)
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .globl _pi
  .data
  .balign 8
_pi:
  .double 3.14
