  .globl _main
  .text
_main:
  pushq  %rbp
  movq   %rsp, %rbp
  movl   $0, %eax
  popq   %rbp
  ret
