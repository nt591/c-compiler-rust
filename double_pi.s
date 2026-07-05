	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0	sdk_version 15, 5
	.globl	_get_pi                         ## -- Begin function get_pi
	.p2align	4, 0x90
_get_pi:                                ## @get_pi
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movsd	_pi(%rip), %xmm0                ## xmm0 = mem[0],zero
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__DATA,__data
	.globl	_pi                             ## @pi
	.p2align	3, 0x0
_pi:
	.quad	0x40091eb851eb851f              ## double 3.1400000000000001

.subsections_via_symbols
