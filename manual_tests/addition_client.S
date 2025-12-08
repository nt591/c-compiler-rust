  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  movl   $1, %edi
  movl   $2, %esi
  call   _add
  movl   %eax, -4(%rbp)
  movl   -4(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
