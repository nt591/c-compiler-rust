  .globl _main
  .text
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $32, %rsp
  movl   $2, -4(%rbp)
  movl   $3, -8(%rbp)
  movl   -8(%rbp), %r10d
  cmpl   %r10d, -4(%rbp)
  movl   $0, %eax
  setl       %al
  movl   %eax, -12(%rbp)
  movl   -12(%rbp), %r10d
  movl   %r10d, -16(%rbp)
  cmpl   $0, -16(%rbp)
  movl   $0, %eax
  sete       %al
  movl   %eax, -20(%rbp)
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
