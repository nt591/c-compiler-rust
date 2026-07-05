  .globl _half
  .text
_half:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $0, %rsp
  movsd  L_const_label.4602678819172646912(%rip), %xmm0
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
  subq   $32, %rsp
  call   _half
  movsd  %xmm0, -8(%rbp)
  movsd  -8(%rbp), %xmm14
  movsd  %xmm14, -16(%rbp)
  movsd  -16(%rbp), %xmm15
  comisd L_const_label.4602678819172646912(%rip), %xmm15
  movl   $0, -20(%rbp)
  setne      -20(%rbp)
  movl   -20(%rbp), %r10d
  movl   %r10d, -24(%rbp)
  cmpl   $0, -24(%rbp)
  je          Lend_label.0
  movl   $1, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.0:
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
  .literal8
  .balign 8
L_const_label.4602678819172646912:
  .double 0.5
