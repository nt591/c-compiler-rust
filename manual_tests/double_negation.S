  .globl _negate
  .text
_negate:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  movsd  %xmm0, -8(%rbp)
  movsd  -8(%rbp), %xmm14
  movsd  %xmm14, -16(%rbp)
  movsd  -16(%rbp), %xmm15
  xorpd  L_const_label_16byte_neg_0_float.9223372036854775808(%rip), %xmm15
  movsd  %xmm15, -16(%rbp)
  movsd  -16(%rbp), %xmm0
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
  movsd  L_const_label.4614253070214989087(%rip), %xmm0
  call   _negate
  movsd  %xmm0, -8(%rbp)
  movsd  -8(%rbp), %xmm14
  movsd  %xmm14, -16(%rbp)
  movsd  L_const_label.4614253070214989087(%rip), %xmm14
  movsd  %xmm14, -24(%rbp)
  movsd  -24(%rbp), %xmm15
  xorpd  L_const_label_16byte_neg_0_float.9223372036854775808(%rip), %xmm15
  movsd  %xmm15, -24(%rbp)
  movsd  -16(%rbp), %xmm15
  comisd -24(%rbp), %xmm15
  movl   $0, -28(%rbp)
  setne      -28(%rbp)
  movl   -28(%rbp), %r10d
  movl   %r10d, -32(%rbp)
  cmpl   $0, -32(%rbp)
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
  .literal16
  .balign 16
L_const_label_16byte_neg_0_float.9223372036854775808:
  .double -0
  .quad 0
  .literal8
  .balign 8
L_const_label.4614253070214989087:
  .double 3.14
