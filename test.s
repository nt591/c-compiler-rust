  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-20, %rsp
  movl   $5, -4(%rbp)
  movl   -4(%rbp), %r11d
                       # BINARY OPERATOR
  imull  $4, %r11d
  movl   %r11d, -4(%rbp)
  movl   -4(%rbp), %eax
  cdq
  movl   $2, %r10d
  idivl  %r10d
  movl   %eax, -8(%rbp)
  movl   $2, -12(%rbp)
                       # BINARY OPERATOR
  addl   $1, -12(%rbp)
  movl   $3, %edx
  cdq
  idivl  -12(%rbp)
  movl   %edx, -16(%rbp)
  movl   -8(%rbp), %r10d
  movl   %r10d, -20(%rbp)
  movl   -16(%rbp), %r10d
                       # BINARY OPERATOR
  subl   %r10d, -20(%rbp)
  movl   -20(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
