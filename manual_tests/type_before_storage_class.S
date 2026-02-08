  .text
_foo:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $0, %rsp
  movl   $3, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .globl _main
  .text
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  call   _foo
  movl   %eax, -4(%rbp)
  movl   -4(%rbp), %r10d
  movl   %r10d, -8(%rbp)
  movl   bar(%rip), %r10d
  addl   %r10d, -8(%rbp)
  movl   -8(%rbp), %eax
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
  .balign 4
_bar:
  .long 4
