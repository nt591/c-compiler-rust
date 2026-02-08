  .globl _main
  .text
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $16, %rsp
  movl   $2, _i.0.decl(%rip)
  movl   $3, _j.1.decl(%rip)
  movl   _j.1.decl(%rip), %r10d
  cmpl   %r10d, _i.0.decl(%rip)
  movl   $0, %eax
  setl       %al
  movl   %eax, -4(%rbp)
  movl   -4(%rbp), %r10d
  movl   %r10d, -8(%rbp)
  cmpl   $0, -8(%rbp)
  movl   $0, %eax
  sete       %al
  movl   %eax, -12(%rbp)
  movl   -12(%rbp), %r10d
  movl   %r10d, -16(%rbp)
  cmpl   $0, -16(%rbp)
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
  .data
  .balign 4
_j.1.decl:
  .long 3
  .data
  .balign 4
_i.0.decl:
  .long 2
