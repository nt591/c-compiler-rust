  .globl _main
  .text
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $352, %rsp
  movsd  _zero(%rip), %xmm14
  movsd  %xmm14, -8(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -8(%rbp), %xmm0
  je          Lend_label.0
  movl   $1, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.0:
  movsd  _rounded_to_zero(%rip), %xmm14
  movsd  %xmm14, -16(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -16(%rbp), %xmm0
  je          Lend_label.1
  movl   $2, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.1:
  movsd  _non_zero(%rip), %xmm14
  movsd  %xmm14, -24(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -24(%rbp), %xmm0
  je          Lelse_label.2
  jmp    Lend_label.3
Lelse_label.2:
  movl   $3, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.3:
  movsd  L_const_label.0(%rip), %xmm14
  movsd  %xmm14, -32(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -32(%rbp), %xmm0
  je          Lend_label.4
  movl   $4, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.4:
  xorpd  %xmm0, %xmm0
  movsd  _non_zero(%rip), %xmm15
  comisd %xmm0, %xmm15
  movl   $0, -36(%rbp)
  sete       -36(%rbp)
  movl   -36(%rbp), %r10d
  movl   %r10d, -40(%rbp)
  cmpl   $0, -40(%rbp)
  je          Lend_label.5
  movl   $4, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.5:
  xorpd  %xmm0, %xmm0
  movsd  _zero(%rip), %xmm15
  comisd %xmm0, %xmm15
  movl   $0, -44(%rbp)
  sete       -44(%rbp)
  cmpl   $0, -44(%rbp)
  movl   $0, -48(%rbp)
  sete       -48(%rbp)
  movl   -48(%rbp), %r10d
  movl   %r10d, -52(%rbp)
  cmpl   $0, -52(%rbp)
  je          Lend_label.6
  movl   $5, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.6:
  xorpd  %xmm0, %xmm0
  movsd  _rounded_to_zero(%rip), %xmm15
  comisd %xmm0, %xmm15
  movl   $0, -56(%rbp)
  sete       -56(%rbp)
  cmpl   $0, -56(%rbp)
  movl   $0, -60(%rbp)
  sete       -60(%rbp)
  movl   -60(%rbp), %r10d
  movl   %r10d, -64(%rbp)
  cmpl   $0, -64(%rbp)
  je          Lend_label.7
  movl   $6, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.7:
  movsd  _non_zero(%rip), %xmm14
  movsd  %xmm14, -72(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -72(%rbp), %xmm0
  je          Land_expr_false.9
  movsd  L_const_label.4607182418800017408(%rip), %xmm14
  movsd  %xmm14, -80(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -80(%rbp), %xmm0
  je          Land_expr_false.9
  movl   $1, -84(%rbp)
  jmp    Land_expr_end.10
Land_expr_false.9:
  movl   $0, -84(%rbp)
Land_expr_end.10:
  cmpl   $0, -84(%rbp)
  movl   $0, -88(%rbp)
  sete       -88(%rbp)
  movl   -88(%rbp), %r10d
  movl   %r10d, -92(%rbp)
  cmpl   $0, -92(%rbp)
  je          Lend_label.8
  movl   $8, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.8:
  movsd  L_const_label.4613937818241073152(%rip), %xmm14
  movsd  %xmm14, -104(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -104(%rbp), %xmm0
  je          Land_expr_false.12
  movsd  _zero(%rip), %xmm14
  movsd  %xmm14, -112(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -112(%rbp), %xmm0
  je          Land_expr_false.12
  movl   $1, -116(%rbp)
  jmp    Land_expr_end.13
Land_expr_false.12:
  movl   $0, -116(%rbp)
Land_expr_end.13:
  movl   -116(%rbp), %r10d
  movl   %r10d, -120(%rbp)
  cmpl   $0, -120(%rbp)
  je          Lend_label.11
  movl   $8, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.11:
  movsd  _rounded_to_zero(%rip), %xmm14
  movsd  %xmm14, -128(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -128(%rbp), %xmm0
  je          Land_expr_false.15
  movsd  L_const_label.4801453603149578240(%rip), %xmm14
  movsd  %xmm14, -136(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -136(%rbp), %xmm0
  je          Land_expr_false.15
  movl   $1, -140(%rbp)
  jmp    Land_expr_end.16
Land_expr_false.15:
  movl   $0, -140(%rbp)
Land_expr_end.16:
  movl   -140(%rbp), %r10d
  movl   %r10d, -144(%rbp)
  cmpl   $0, -144(%rbp)
  je          Lend_label.14
  movl   $9, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.14:
  movq   $18446744073709551615, %r10
  movq   %r10, -152(%rbp)
  cmpq   $0, -152(%rbp)
  je          Land_expr_false.18
  movsd  _zero(%rip), %xmm14
  movsd  %xmm14, -160(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -160(%rbp), %xmm0
  je          Land_expr_false.18
  movl   $1, -164(%rbp)
  jmp    Land_expr_end.19
Land_expr_false.18:
  movl   $0, -164(%rbp)
Land_expr_end.19:
  movl   -164(%rbp), %r10d
  movl   %r10d, -168(%rbp)
  cmpl   $0, -168(%rbp)
  je          Lend_label.17
  movl   $10, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.17:
  movsd  _non_zero(%rip), %xmm14
  movsd  %xmm14, -176(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -176(%rbp), %xmm0
  je          Land_expr_false.21
  movq   $5, -184(%rbp)
  cmpq   $0, -184(%rbp)
  je          Land_expr_false.21
  movl   $1, -188(%rbp)
  jmp    Land_expr_end.22
Land_expr_false.21:
  movl   $0, -188(%rbp)
Land_expr_end.22:
  cmpl   $0, -188(%rbp)
  movl   $0, -192(%rbp)
  sete       -192(%rbp)
  movl   -192(%rbp), %r10d
  movl   %r10d, -196(%rbp)
  cmpl   $0, -196(%rbp)
  je          Lend_label.20
  movl   $11, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.20:
  movsd  L_const_label.4617315517961601024(%rip), %xmm14
  movsd  %xmm14, -208(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -208(%rbp), %xmm0
  jne         Lor_expr_true.24
  movsd  _zero(%rip), %xmm14
  movsd  %xmm14, -216(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -216(%rbp), %xmm0
  jne         Lor_expr_true.24
  movl   $0, -220(%rbp)
  jmp    Lor_expr_end.25
Lor_expr_true.24:
  movl   $1, -220(%rbp)
Lor_expr_end.25:
  cmpl   $0, -220(%rbp)
  movl   $0, -224(%rbp)
  sete       -224(%rbp)
  movl   -224(%rbp), %r10d
  movl   %r10d, -228(%rbp)
  cmpl   $0, -228(%rbp)
  je          Lend_label.23
  movl   $12, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.23:
  movsd  _zero(%rip), %xmm14
  movsd  %xmm14, -240(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -240(%rbp), %xmm0
  jne         Lor_expr_true.27
  movsd  _rounded_to_zero(%rip), %xmm14
  movsd  %xmm14, -248(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -248(%rbp), %xmm0
  jne         Lor_expr_true.27
  movl   $0, -252(%rbp)
  jmp    Lor_expr_end.28
Lor_expr_true.27:
  movl   $1, -252(%rbp)
Lor_expr_end.28:
  movl   -252(%rbp), %r10d
  movl   %r10d, -256(%rbp)
  cmpl   $0, -256(%rbp)
  je          Lend_label.26
  movl   $13, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.26:
  movsd  _rounded_to_zero(%rip), %xmm14
  movsd  %xmm14, -264(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -264(%rbp), %xmm0
  jne         Lor_expr_true.30
  movsd  L_const_label.4547007122018943789(%rip), %xmm14
  movsd  %xmm14, -272(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -272(%rbp), %xmm0
  jne         Lor_expr_true.30
  movl   $0, -276(%rbp)
  jmp    Lor_expr_end.31
Lor_expr_true.30:
  movl   $1, -276(%rbp)
Lor_expr_end.31:
  cmpl   $0, -276(%rbp)
  movl   $0, -280(%rbp)
  sete       -280(%rbp)
  movl   -280(%rbp), %r10d
  movl   %r10d, -284(%rbp)
  cmpl   $0, -284(%rbp)
  je          Lend_label.29
  movl   $14, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.29:
  movsd  _non_zero(%rip), %xmm14
  movsd  %xmm14, -296(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -296(%rbp), %xmm0
  jne         Lor_expr_true.33
  movl   $0, -300(%rbp)
  cmpl   $0, -300(%rbp)
  jne         Lor_expr_true.33
  movl   $0, -304(%rbp)
  jmp    Lor_expr_end.34
Lor_expr_true.33:
  movl   $1, -304(%rbp)
Lor_expr_end.34:
  cmpl   $0, -304(%rbp)
  movl   $0, -308(%rbp)
  sete       -308(%rbp)
  movl   -308(%rbp), %r10d
  movl   %r10d, -312(%rbp)
  cmpl   $0, -312(%rbp)
  je          Lend_label.32
  movl   $15, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.32:
  movl   $0, -316(%rbp)
  cmpl   $0, -316(%rbp)
  jne         Lor_expr_true.36
  movsd  L_const_label.4512825593480736141(%rip), %xmm14
  movsd  %xmm14, -328(%rbp)
  xorpd  %xmm0, %xmm0
  comisd -328(%rbp), %xmm0
  jne         Lor_expr_true.36
  movl   $0, -332(%rbp)
  jmp    Lor_expr_end.37
Lor_expr_true.36:
  movl   $1, -332(%rbp)
Lor_expr_end.37:
  cmpl   $0, -332(%rbp)
  movl   $0, -336(%rbp)
  sete       -336(%rbp)
  movl   -336(%rbp), %r10d
  movl   %r10d, -340(%rbp)
  cmpl   $0, -340(%rbp)
  je          Lend_label.35
  movl   $16, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.35:
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
  .globl _rounded_to_zero
  .data
  .balign 8
_rounded_to_zero:
  .double 0.0
  .globl _non_zero
  .data
  .balign 8
_non_zero:
  .double 1e-20
  .globl _one
  .data
  .balign 8
_one:
  .double 1.0
  .globl _zero
  .data
  .balign 8
_zero:
  .double 0.0
  .literal8
  .balign 8
L_const_label.4617315517961601024:
  .double 5.0
  .literal8
  .balign 8
L_const_label.4512825593480736141:
  .double 5e-7
  .literal8
  .balign 8
L_const_label.4547007122018943789:
  .double 0.0001
  .literal8
  .balign 8
L_const_label.4607182418800017408:
  .double 1.0
  .literal8
  .balign 8
L_const_label.0:
  .double 0.0
  .literal8
  .balign 8
L_const_label.4801453603149578240:
  .double 10000000000000.0
  .literal8
  .balign 8
L_const_label.4613937818241073152:
  .double 3.0
