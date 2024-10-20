  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-16, %rsp
  movl   $5, -4(%rbp)
  movl   -4(%rbp), %r11d
  imull  $4, %r11d
  movl   %r11d, -4(%rbp)
  movl   $4, -8(%rbp)
  subl   $5, -8(%rbp)
  movl   -8(%rbp), %r10d
  movl   %r10d, -12(%rbp)
  andl   $6, -12(%rbp)
  movl   -4(%rbp), %r10d
  movl   %r10d, -16(%rbp)
  movl   -12(%rbp), %r10d
  orl    %r10d, -16(%rbp)
  movl   -16(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
