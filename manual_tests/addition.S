  .globl _add
_add:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  movl   %edi, -4(%rbp)
  movl   %esi, -8(%rbp)
  movl   -4(%rbp), %r10d
  movl   %r10d, -12(%rbp)
  movl   -8(%rbp), %r10d
  addl   %r10d, -12(%rbp)
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
