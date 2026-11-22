#ifndef __gb100_clc96f_h__
#define __gb100_clc96f_h__

#define  BLACKWELL_CHANNEL_GPFIFO_A                           (0x0000C96F)

typedef volatile struct Nvc96fControl_struct {
 NvU32 Ignored00[0x23];
 NvU32 GPPut;
 NvU32 Ignored01[0x5c];
} Nvc96fControl, BlackwellAControlGPFifo;

#define NVC96F_SET_OBJECT                                          (0x00000000)
#define NVC96F_MEM_OP_A                                            (0x00000028)
#define NVC96F_MEM_OP_A_TLB_INVALIDATE_INVALIDATION_SIZE                   5:0
#define NVC96F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE                         7:6
#define NVC96F_MEM_OP_A_TLB_INVALIDATE_INVAL_SCOPE_NON_LINK_TLBS             2
#define NVC96F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR                         11:11
#define NVC96F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR_EN                 0x00000001
#define NVC96F_MEM_OP_A_TLB_INVALIDATE_SYSMEMBAR_DIS                0x00000000
#define NVC96F_MEM_OP_A_TLB_INVALIDATE_TARGET_ADDR_LO                    31:12
#define NVC96F_MEM_OP_B                                            (0x0000002c)
#define NVC96F_MEM_OP_B_TLB_INVALIDATE_TARGET_ADDR_HI                     31:0
#define NVC96F_MEM_OP_C                                            (0x00000030)
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PDB                                 0:0
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PDB_ONE                      0x00000000
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PDB_ALL                      0x00000001
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_GPC                                 1:1
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_GPC_ENABLE                   0x00000000
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_GPC_DISABLE                  0x00000001
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_REPLAY                              4:2
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_REPLAY_NONE                  0x00000000
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE                            6:5
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_NONE                0x00000000
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_ACK_TYPE_GLOBALLY            0x00000001
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL                    9:7
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_ALL         0x00000000
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_PTE_ONLY    0x00000001
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PAGE_TABLE_LEVEL_UP_TO_PDE4  0x00000006
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE                          11:10
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE_VID_MEM             0x00000000
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PDB_APERTURE_SYS_MEM_COHERENT    0x00000002
#define NVC96F_MEM_OP_C_TLB_INVALIDATE_PDB_ADDR_LO                       31:12

#define NVC96F_MEM_OP_D                                            (0x00000034)
#define NVC96F_MEM_OP_D_TLB_INVALIDATE_PDB_ADDR_HI                        26:0
#define NVC96F_MEM_OP_D_OPERATION                                        31:27
#define NVC96F_MEM_OP_D_OPERATION_MMU_TLB_INVALIDATE                0x00000009
#define NVC96F_MEM_OP_D_OPERATION_MMU_TLB_INVALIDATE_TARGETED       0x0000000a
#define NVC96F_MEM_OP_D_OPERATION_L2_FLUSH_DIRTY                    0x00000010
#define NVC96F_MEM_OP_D_OPERATION_L2_SYSMEM_NCOH_INVALIDATE         0x00000011

#define NVC96F_SEM_ADDR_LO                                         (0x0000005c)
#define NVC96F_SEM_ADDR_LO_OFFSET                                         31:2
#define NVC96F_SEM_ADDR_HI                                         (0x00000060)
#define NVC96F_SEM_ADDR_HI_OFFSET                                         24:0
#define NVC96F_SEM_PAYLOAD_LO                                      (0x00000064)
#define NVC96F_SEM_PAYLOAD_HI                                      (0x00000068)
#define NVC96F_SEM_EXECUTE                                         (0x0000006c)
#define NVC96F_SEM_EXECUTE_OPERATION                                       2:0
#define NVC96F_SEM_EXECUTE_OPERATION_ACQUIRE                        0x00000000
#define NVC96F_SEM_EXECUTE_OPERATION_RELEASE                        0x00000001
#define NVC96F_SEM_EXECUTE_RELEASE_WFI                                   20:20
#define NVC96F_SEM_EXECUTE_RELEASE_WFI_DIS                          0x00000000
#define NVC96F_SEM_EXECUTE_PAYLOAD_SIZE                                  24:24
#define NVC96F_SEM_EXECUTE_PAYLOAD_SIZE_32BIT                       0x00000000

#define NVC96F_GP_ENTRY__SIZE                                          8
#define NVC96F_GP_ENTRY0_FETCH                                       0:0
#define NVC96F_GP_ENTRY0_FETCH_UNCONDITIONAL                  0x00000000
#define NVC96F_GP_ENTRY0_FETCH_CONDITIONAL                    0x00000001
#define NVC96F_GP_ENTRY0_GET                                        31:2
#define NVC96F_GP_ENTRY0_OPERAND                                    31:0
#define NVC96F_GP_ENTRY0_PB_EXTENDED_BASE_OPERAND                   24:8
#define NVC96F_GP_ENTRY1_GET_HI                                      7:0
#define NVC96F_GP_ENTRY1_LEVEL                                       9:9
#define NVC96F_GP_ENTRY1_LEVEL_MAIN                           0x00000000
#define NVC96F_GP_ENTRY1_LEVEL_SUBROUTINE                     0x00000001
#define NVC96F_GP_ENTRY1_LENGTH                                    30:10
#define NVC96F_GP_ENTRY1_SYNC                                      31:31
#define NVC96F_GP_ENTRY1_SYNC_PROCEED                         0x00000000
#define NVC96F_GP_ENTRY1_SYNC_WAIT                            0x00000001
#define NVC96F_GP_ENTRY1_OPCODE                                      7:0
#define NVC96F_GP_ENTRY1_OPCODE_NOP                           0x00000000
#define NVC96F_GP_ENTRY1_OPCODE_ILLEGAL                       0x00000001
#define NVC96F_GP_ENTRY1_OPCODE_GP_CRC                        0x00000002
#define NVC96F_GP_ENTRY1_OPCODE_PB_CRC                        0x00000003
#define NVC96F_GP_ENTRY1_OPCODE_SET_PB_SEGMENT_EXTENDED_BASE  0x00000004

#endif
