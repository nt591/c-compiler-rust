  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-12, %rsp
  movl   $1, -4(%rbp)
  negl   -4(%rbp)
  movl   $2, -8(%rbp)
  negl   -8(%rbp)
  movl   -8(%rbp), %r10d
  cmpl   %r10d, -4(%rbp)
  movl   $0, -12(%rbp)
  setle       -12(%rbp)
  movl   -12(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
