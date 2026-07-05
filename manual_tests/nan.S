  .globl _main
  .text
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $320, %rsp
  movsd  L_const_label.0(%rip), %xmm14
  movsd  %xmm14, -8(%rbp)
  movsd  -8(%rbp), %xmm15
  divsd  _zero.1.decl(%rip), %xmm15
  movsd  %xmm15, -8(%rbp)
  movsd  -8(%rbp), %xmm14
  movsd  %xmm14, -16(%rbp)
  movsd  -16(%rbp), %xmm15
  comisd L_const_label.0(%rip), %xmm15
  setb       %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -20(%rbp)
  movl   -20(%rbp), %r10d
  movl   %r10d, -24(%rbp)
  cmpl   $0, -24(%rbp)
  jne         Lor_expr_true.7
  movsd  -16(%rbp), %xmm15
  comisd L_const_label.0(%rip), %xmm15
  sete       %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -28(%rbp)
  movl   -28(%rbp), %r10d
  movl   %r10d, -32(%rbp)
  cmpl   $0, -32(%rbp)
  jne         Lor_expr_true.7
  movl   $0, -36(%rbp)
  jmp    Lor_expr_end.8
Lor_expr_true.7:
  movl   $1, -36(%rbp)
Lor_expr_end.8:
  movl   -36(%rbp), %r10d
  movl   %r10d, -40(%rbp)
  cmpl   $0, -40(%rbp)
  jne         Lor_expr_true.5
  movsd  -16(%rbp), %xmm15
  comisd L_const_label.0(%rip), %xmm15
  seta       %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -44(%rbp)
  movl   -44(%rbp), %r10d
  movl   %r10d, -48(%rbp)
  cmpl   $0, -48(%rbp)
  jne         Lor_expr_true.5
  movl   $0, -52(%rbp)
  jmp    Lor_expr_end.6
Lor_expr_true.5:
  movl   $1, -52(%rbp)
Lor_expr_end.6:
  movl   -52(%rbp), %r10d
  movl   %r10d, -56(%rbp)
  cmpl   $0, -56(%rbp)
  jne         Lor_expr_true.3
  movsd  -16(%rbp), %xmm15
  comisd L_const_label.0(%rip), %xmm15
  setbe      %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -60(%rbp)
  movl   -60(%rbp), %r10d
  movl   %r10d, -64(%rbp)
  cmpl   $0, -64(%rbp)
  jne         Lor_expr_true.3
  movl   $0, -68(%rbp)
  jmp    Lor_expr_end.4
Lor_expr_true.3:
  movl   $1, -68(%rbp)
Lor_expr_end.4:
  movl   -68(%rbp), %r10d
  movl   %r10d, -72(%rbp)
  cmpl   $0, -72(%rbp)
  jne         Lor_expr_true.1
  movsd  -16(%rbp), %xmm15
  comisd L_const_label.0(%rip), %xmm15
  setae      %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -76(%rbp)
  movl   -76(%rbp), %r10d
  movl   %r10d, -80(%rbp)
  cmpl   $0, -80(%rbp)
  jne         Lor_expr_true.1
  movl   $0, -84(%rbp)
  jmp    Lor_expr_end.2
Lor_expr_true.1:
  movl   $1, -84(%rbp)
Lor_expr_end.2:
  movl   -84(%rbp), %r10d
  movl   %r10d, -88(%rbp)
  cmpl   $0, -88(%rbp)
  je          Lend_label.0
  movl   $1, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.0:
  movl   $1, %r10d
  cvtsi2sdl  %r10d, %xmm15
  movsd  %xmm15, -96(%rbp)
  movsd  -96(%rbp), %xmm15
  comisd -16(%rbp), %xmm15
  seta       %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -100(%rbp)
  movl   -100(%rbp), %r10d
  movl   %r10d, -104(%rbp)
  cmpl   $0, -104(%rbp)
  jne         Lor_expr_true.16
  movl   $1, %r10d
  cvtsi2sdl  %r10d, %xmm15
  movsd  %xmm15, -112(%rbp)
  movsd  -112(%rbp), %xmm15
  comisd -16(%rbp), %xmm15
  sete       %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -116(%rbp)
  movl   -116(%rbp), %r10d
  movl   %r10d, -120(%rbp)
  cmpl   $0, -120(%rbp)
  jne         Lor_expr_true.16
  movl   $0, -124(%rbp)
  jmp    Lor_expr_end.17
Lor_expr_true.16:
  movl   $1, -124(%rbp)
Lor_expr_end.17:
  movl   -124(%rbp), %r10d
  movl   %r10d, -128(%rbp)
  cmpl   $0, -128(%rbp)
  jne         Lor_expr_true.14
  movl   $1, %r10d
  cvtsi2sdl  %r10d, %xmm15
  movsd  %xmm15, -136(%rbp)
  movsd  -136(%rbp), %xmm15
  comisd -16(%rbp), %xmm15
  seta       %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -140(%rbp)
  movl   -140(%rbp), %r10d
  movl   %r10d, -144(%rbp)
  cmpl   $0, -144(%rbp)
  jne         Lor_expr_true.14
  movl   $0, -148(%rbp)
  jmp    Lor_expr_end.15
Lor_expr_true.14:
  movl   $1, -148(%rbp)
Lor_expr_end.15:
  movl   -148(%rbp), %r10d
  movl   %r10d, -152(%rbp)
  cmpl   $0, -152(%rbp)
  jne         Lor_expr_true.12
  movl   $1, %r10d
  cvtsi2sdl  %r10d, %xmm15
  movsd  %xmm15, -160(%rbp)
  movsd  -160(%rbp), %xmm15
  comisd -16(%rbp), %xmm15
  setbe      %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -164(%rbp)
  movl   -164(%rbp), %r10d
  movl   %r10d, -168(%rbp)
  cmpl   $0, -168(%rbp)
  jne         Lor_expr_true.12
  movl   $0, -172(%rbp)
  jmp    Lor_expr_end.13
Lor_expr_true.12:
  movl   $1, -172(%rbp)
Lor_expr_end.13:
  movl   -172(%rbp), %r10d
  movl   %r10d, -176(%rbp)
  cmpl   $0, -176(%rbp)
  jne         Lor_expr_true.10
  movl   $1, %r10d
  cvtsi2sdl  %r10d, %xmm15
  movsd  %xmm15, -184(%rbp)
  movsd  -184(%rbp), %xmm15
  comisd -16(%rbp), %xmm15
  setae      %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -188(%rbp)
  movl   -188(%rbp), %r10d
  movl   %r10d, -192(%rbp)
  cmpl   $0, -192(%rbp)
  jne         Lor_expr_true.10
  movl   $0, -196(%rbp)
  jmp    Lor_expr_end.11
Lor_expr_true.10:
  movl   $1, -196(%rbp)
Lor_expr_end.11:
  movl   -196(%rbp), %r10d
  movl   %r10d, -200(%rbp)
  cmpl   $0, -200(%rbp)
  je          Lend_label.9
  movl   $2, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.9:
  movsd  -16(%rbp), %xmm15
  comisd -16(%rbp), %xmm15
  sete       %r10b
  setnp      %r11b
  andl   %r10d, %r11d
  movl   %r11d, -204(%rbp)
  movl   -204(%rbp), %r10d
  movl   %r10d, -208(%rbp)
  cmpl   $0, -208(%rbp)
  je          Lend_label.18
  movl   $3, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.18:
  movsd  -16(%rbp), %xmm15
  comisd -16(%rbp), %xmm15
  setne      %r10b
  setp       %r11b
  orl    %r10d, %r11d
  movl   %r11d, -212(%rbp)
  cmpl   $0, -212(%rbp)
  movl   $0, -216(%rbp)
  sete       -216(%rbp)
  movl   -216(%rbp), %r10d
  movl   %r10d, -220(%rbp)
  cmpl   $0, -220(%rbp)
  je          Lend_label.19
  movl   $4, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.19:
  movsd  -16(%rbp), %xmm0
  call   _double_isnan
  movl   %eax, -224(%rbp)
  cmpl   $0, -224(%rbp)
  movl   $0, -228(%rbp)
  sete       -228(%rbp)
  movl   -228(%rbp), %r10d
  movl   %r10d, -232(%rbp)
  cmpl   $0, -232(%rbp)
  je          Lend_label.20
  movl   $5, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.20:
  movl   $4, %r10d
  cvtsi2sdl  %r10d, %xmm15
  movsd  %xmm15, -240(%rbp)
  movsd  -240(%rbp), %xmm14
  movsd  %xmm14, -248(%rbp)
  movsd  -248(%rbp), %xmm15
  mulsd  -16(%rbp), %xmm15
  movsd  %xmm15, -248(%rbp)
  movsd  -248(%rbp), %xmm0
  call   _double_isnan
  movl   %eax, -252(%rbp)
  cmpl   $0, -252(%rbp)
  movl   $0, -256(%rbp)
  sete       -256(%rbp)
  movl   -256(%rbp), %r10d
  movl   %r10d, -260(%rbp)
  cmpl   $0, -260(%rbp)
  je          Lend_label.21
  movl   $6, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.21:
  movsd  L_const_label.4657056266235936768(%rip), %xmm14
  movsd  %xmm14, -272(%rbp)
  movsd  -272(%rbp), %xmm15
  divsd  -16(%rbp), %xmm15
  movsd  %xmm15, -272(%rbp)
  movsd  -272(%rbp), %xmm0
  call   _double_isnan
  movl   %eax, -276(%rbp)
  cmpl   $0, -276(%rbp)
  movl   $0, -280(%rbp)
  sete       -280(%rbp)
  movl   -280(%rbp), %r10d
  movl   %r10d, -284(%rbp)
  cmpl   $0, -284(%rbp)
  je          Lend_label.22
  movl   $7, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.22:
  movsd  -16(%rbp), %xmm14
  movsd  %xmm14, -296(%rbp)
  movsd  -296(%rbp), %xmm15
  xorpd  L_const_label_16byte_neg_0_float.9223372036854775808(%rip), %xmm15
  movsd  %xmm15, -296(%rbp)
  movsd  -296(%rbp), %xmm0
  call   _double_isnan
  movl   %eax, -300(%rbp)
  cmpl   $0, -300(%rbp)
  movl   $0, -304(%rbp)
  sete       -304(%rbp)
  movl   -304(%rbp), %r10d
  movl   %r10d, -308(%rbp)
  cmpl   $0, -308(%rbp)
  je          Lend_label.23
  movl   $8, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
Lend_label.23:
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
  .data
  .balign 8
_zero.1.decl:
  .double 0.0
  .literal16
  .balign 16
L_const_label_16byte_neg_0_float.9223372036854775808:
  .double -0.0
  .quad 0
  .literal8
  .balign 8
L_const_label.0:
  .double 0.0
  .literal8
  .balign 8
L_const_label.4657056266235936768:
  .double 2200.0
