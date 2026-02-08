  .globl _print_alphabet
  .text
_print_alphabet:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $32, %rsp
  movl   $0, _count.1.decl(%rip)
  movl   _count.1.decl(%rip), %r10d
  movl   %r10d, -4(%rbp)
  addl   $65, -4(%rbp)
  movl   -4(%rbp), %edi
  call   _putchar
  movl   %eax, -8(%rbp)
  movl   _count.1.decl(%rip), %r10d
  movl   %r10d, -12(%rbp)
  addl   $1, -12(%rbp)
  movl   -12(%rbp), %r10d
  movl   %r10d, _count.1.decl(%rip)
  cmpl   $26, _count.1.decl(%rip)
  movl   $0, -16(%rbp)
  setl       -16(%rbp)
  movl   -16(%rbp), %r10d
  movl   %r10d, -20(%rbp)
  cmpl   $0, -20(%rbp)
  je          Lend_label.0
  call   _print_alphabet
  movl   %eax, -24(%rbp)
Lend_label.0:
  movl   _count.1.decl(%rip), %eax
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
  call   _print_alphabet
  movl   %eax, -4(%rbp)
  movl   $0, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
  .bss
  .balign 4
_count.1.decl:
  .zero 4
