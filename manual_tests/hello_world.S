  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $64, %rsp
  movl   $72, %edi
  call   _putchar
  movl   %eax, -4(%rbp)
  movl   $101, %edi
  call   _putchar
  movl   %eax, -8(%rbp)
  movl   $108, %edi
  call   _putchar
  movl   %eax, -12(%rbp)
  movl   $108, %edi
  call   _putchar
  movl   %eax, -16(%rbp)
  movl   $111, %edi
  call   _putchar
  movl   %eax, -20(%rbp)
  movl   $44, %edi
  call   _putchar
  movl   %eax, -24(%rbp)
  movl   $32, %edi
  call   _putchar
  movl   %eax, -28(%rbp)
  movl   $87, %edi
  call   _putchar
  movl   %eax, -32(%rbp)
  movl   $111, %edi
  call   _putchar
  movl   %eax, -36(%rbp)
  movl   $114, %edi
  call   _putchar
  movl   %eax, -40(%rbp)
  movl   $108, %edi
  call   _putchar
  movl   %eax, -44(%rbp)
  movl   $100, %edi
  call   _putchar
  movl   %eax, -48(%rbp)
  movl   $33, %edi
  call   _putchar
  movl   %eax, -52(%rbp)
  movl   $10, %edi
  call   _putchar
  movl   %eax, -56(%rbp)
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
