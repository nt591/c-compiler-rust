  .globl _addition
  .text
_addition:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  movsd  _point_one(%rip), %xmm14
  movsd  %xmm14, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  addsd  _point_two(%rip), %xmm15
  movsd  %xmm15, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  comisd L_const_label.4599075939470750516(%rip), %xmm15
  movl   $0, -12(%rbp)
  jp          Lquiet_nan____internal.0
  sete       -12(%rbp)
Lquiet_nan____internal.0:
  movl   -12(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .globl _subtraction
  .text
_subtraction:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  movsd  _four(%rip), %xmm14
  movsd  %xmm14, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  subsd  L_const_label.4607182418800017408(%rip), %xmm15
  movsd  %xmm15, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  comisd L_const_label.4613937818241073152(%rip), %xmm15
  movl   $0, -12(%rbp)
  jp          Lquiet_nan____internal.1
  sete       -12(%rbp)
Lquiet_nan____internal.1:
  movl   -12(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .globl _multiplication
  .text
_multiplication:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  movsd  L_const_label.4576918229304087675(%rip), %xmm14
  movsd  %xmm14, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  mulsd  _point_three(%rip), %xmm15
  movsd  %xmm15, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  comisd L_const_label.4569063951553953530(%rip), %xmm15
  movl   $0, -12(%rbp)
  jp          Lquiet_nan____internal.2
  sete       -12(%rbp)
Lquiet_nan____internal.2:
  movl   -12(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .globl _division
  .text
_division:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  movsd  L_const_label.4619567317775286272(%rip), %xmm14
  movsd  %xmm14, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  divsd  _two(%rip), %xmm15
  movsd  %xmm15, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  comisd L_const_label.4615063718147915776(%rip), %xmm15
  movl   $0, -12(%rbp)
  jp          Lquiet_nan____internal.3
  sete       -12(%rbp)
Lquiet_nan____internal.3:
  movl   -12(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .globl _negation
  .text
_negation:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $32, %rsp
  movsd  _twelveE30(%rip), %xmm14
  movsd  %xmm14, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  xorpd  L_const_label_16byte_neg_0_float.9223372036854775808(%rip), %xmm15
  movsd  %xmm15, -8(%rbp)
  movsd  -8(%rbp), %xmm14
  movsd  %xmm14, -16(%rbp)
  movsd  L_const_label.5071878651310008751(%rip), %xmm14
  movsd  %xmm14, -24(%rbp)
  movsd  -24(%rbp), %xmm15
  addsd  -16(%rbp), %xmm15
  movsd  %xmm15, -24(%rbp)
  xorpd  %xmm0, %xmm0
  movsd  -24(%rbp), %xmm15
  comisd %xmm0, %xmm15
  movl   $0, -28(%rbp)
  jp          Lquiet_nan____internal.4
  sete       -28(%rbp)
Lquiet_nan____internal.4:
  movl   -28(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .globl _complex_expression
  .text
_complex_expression:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $48, %rsp
  movsd  _two(%rip), %xmm14
  movsd  %xmm14, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  addsd  _three(%rip), %xmm15
  movsd  %xmm15, -8(%rbp)
  movsd  L_const_label.4638672431819522048(%rip), %xmm14
  movsd  %xmm14, -16(%rbp)
  movsd  -16(%rbp), %xmm15
  mulsd  _four(%rip), %xmm15
  movsd  %xmm15, -16(%rbp)
  movsd  -8(%rbp), %xmm14
  movsd  %xmm14, -24(%rbp)
  movsd  -24(%rbp), %xmm15
  subsd  -16(%rbp), %xmm15
  movsd  %xmm15, -24(%rbp)
  movsd  -24(%rbp), %xmm14
  movsd  %xmm14, -32(%rbp)
  movsd  L_const_label.4647591670144040960(%rip), %xmm14
  movsd  %xmm14, -40(%rbp)
  movsd  -40(%rbp), %xmm15
  xorpd  L_const_label_16byte_neg_0_float.9223372036854775808(%rip), %xmm15
  movsd  %xmm15, -40(%rbp)
  movsd  -32(%rbp), %xmm15
  comisd -40(%rbp), %xmm15
  movl   $0, -44(%rbp)
  jp          Lquiet_nan____internal.5
  sete       -44(%rbp)
Lquiet_nan____internal.5:
  movl   -44(%rbp), %eax
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
  subq   $80, %rsp
  call   _addition
  movl   %eax, -4(%rbp)
  cmpl   $0, -4(%rbp)
  movl   $0, -8(%rbp)
  sete       -8(%rbp)
  movl   -8(%rbp), %r10d
  movl   %r10d, -12(%rbp)
  cmpl   $0, -12(%rbp)
  je          Lend_label.0
  movl   $1, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.0:
  call   _subtraction
  movl   %eax, -16(%rbp)
  cmpl   $0, -16(%rbp)
  movl   $0, -20(%rbp)
  sete       -20(%rbp)
  movl   -20(%rbp), %r10d
  movl   %r10d, -24(%rbp)
  cmpl   $0, -24(%rbp)
  je          Lend_label.1
  movl   $2, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.1:
  call   _multiplication
  movl   %eax, -28(%rbp)
  cmpl   $0, -28(%rbp)
  movl   $0, -32(%rbp)
  sete       -32(%rbp)
  movl   -32(%rbp), %r10d
  movl   %r10d, -36(%rbp)
  cmpl   $0, -36(%rbp)
  je          Lend_label.2
  movl   $3, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.2:
  call   _division
  movl   %eax, -40(%rbp)
  cmpl   $0, -40(%rbp)
  movl   $0, -44(%rbp)
  sete       -44(%rbp)
  movl   -44(%rbp), %r10d
  movl   %r10d, -48(%rbp)
  cmpl   $0, -48(%rbp)
  je          Lend_label.3
  movl   $4, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.3:
  call   _negation
  movl   %eax, -52(%rbp)
  cmpl   $0, -52(%rbp)
  movl   $0, -56(%rbp)
  sete       -56(%rbp)
  movl   -56(%rbp), %r10d
  movl   %r10d, -60(%rbp)
  cmpl   $0, -60(%rbp)
  je          Lend_label.4
  movl   $5, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.4:
  call   _complex_expression
  movl   %eax, -64(%rbp)
  cmpl   $0, -64(%rbp)
  movl   $0, -68(%rbp)
  sete       -68(%rbp)
  movl   -68(%rbp), %r10d
  movl   %r10d, -72(%rbp)
  cmpl   $0, -72(%rbp)
  je          Lend_label.5
  movl   $5, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.5:
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
  .globl _three
  .data
  .balign 8
_three:
  .double 3.0
  .globl _two
  .data
  .balign 8
_two:
  .double 2.0
  .globl _point_three
  .data
  .balign 8
_point_three:
  .double 0.3
  .globl _point_two
  .data
  .balign 8
_point_two:
  .double 0.2
  .globl _four
  .data
  .balign 8
_four:
  .double 4.0
  .globl _twelveE30
  .data
  .balign 8
_twelveE30:
  .double 1.2e31
  .globl _point_one
  .data
  .balign 8
_point_one:
  .double 0.1
  .literal8
  .balign 8
L_const_label.4607182418800017408:
  .double 1.0
  .literal8
  .balign 8
L_const_label.4599075939470750516:
  .double 0.30000000000000004
  .literal8
  .balign 8
L_const_label.4638672431819522048:
  .double 127.5
  .literal8
  .balign 8
L_const_label.4576918229304087675:
  .double 0.01
  .literal8
  .balign 8
L_const_label.4619567317775286272:
  .double 7.0
  .literal8
  .balign 8
L_const_label.4615063718147915776:
  .double 3.5
  .literal8
  .balign 8
L_const_label.4613937818241073152:
  .double 3.0
  .literal8
  .balign 8
L_const_label.4569063951553953530:
  .double 0.003
  .literal16
  .balign 16
L_const_label_16byte_neg_0_float.9223372036854775808:
  .double -0.0
  .quad 0
  .literal8
  .balign 8
L_const_label.5071878651310008751:
  .double 1.2e31
  .literal8
  .balign 8
L_const_label.4647591670144040960:
  .double 505.0
