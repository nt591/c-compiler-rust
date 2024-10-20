  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-8, %rsp
  movl   $5, -4(%rbp)
  negl   -4(%rbp)
  movl   -4(%rbp), %eax
  cdq
  shrl   $30, %eax
  orl   %edx, %eax
  movl   %eax, -8(%rbp)
  movl   -8(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
