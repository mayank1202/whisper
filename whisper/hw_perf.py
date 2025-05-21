import numpy as np
import torch

MEM_BW_L2_MBps = (256*1024) #256 GB/s
DTYPE_BITS = 8
FREQ_MHz = 1250

MATRIX_AVAILABLE = False
VECTOR_AVAILABLE = True

VECTOR_DLEN_BITS = 256
NUM_VECTOR_OPS_PER_CYCLE = VECTOR_DLEN_BITS//DTYPE_BITS

NUM_LAYERNORM_OPS = 10
NUM_SOFTMAX_OPS = 10

def volume(tensor):
    vol=1
    for dim in tensor.shape:
        vol *= dim
    return vol


class PerfKPIs():
    def __init__(self):
        self.num_ops = 0
        self.bytes_read = 0
        self.bytes_written = 0
        self.execution_time_us = 0

    def reset(self):
        self.num_ops = 0
        self.bytes_read = 0
        self.bytes_written = 0
        self.execution_time_us = 0

    def add(self, another_perf_kpi):
        self.num_ops += another_perf_kpi.num_ops
        self.bytes_read += another_perf_kpi.bytes_read
        self.bytes_written += another_perf_kpi.bytes_written
        self.execution_time_us += another_perf_kpi.execution_time_us
    
    def disp(self):
        print('self.num_ops = ',self.num_ops)
        print('self.bytes_read = ',self.bytes_read)
        print('self.bytes_written = ',self.bytes_written)
        print('self.execution_time_us = ',self.execution_time_us)

#Matrix multiplication of two matrices
def matmul_2d_perf_kpis(mat1, mat2):
    m, n1 = mat1.shape
    n2, k = mat2.shape

    mm_perf_kpis = PerfKPIs()

    if(n1!=n2):
        print('matmul_2d_perf_kpis : dimension mismatch ', mat1.shape, mat2.shape)
        return None

    n = n1    
    mm_perf_kpis.num_ops = 2*m*n*k

    if(MATRIX_AVAILABLE == True):
        print('TBD')
        return None
    elif(VECTOR_AVAILABLE == True):     
        
        mm_perf_kpis.bytes_read = DTYPE_BITS * ((m*n) + (n*k))
        mm_perf_kpis.bytes_written = DTYPE_BITS * (m*k)
        memory_time = (mm_perf_kpis.bytes_read + mm_perf_kpis.bytes_written)/MEM_BW_L2_MBps

        compute_cycles = 2*m*k * (n+NUM_VECTOR_OPS_PER_CYCLE-1)//NUM_VECTOR_OPS_PER_CYCLE
        compute_time = compute_cycles/FREQ_MHz

        mm_perf_kpis.execution_time_us = max(memory_time, compute_time)

        return mm_perf_kpis   
    else:     
        print('TBD')
        return None


#1d convolution
def conv1d_perf_kpis(in_tensor, conv_kernel):
    c1d_perf_kpis = PerfKPIs()

    batchsize, length_in, c_in = in_tensor.shape
    length_out = (length_in + conv_kernel.padding[0] - 1)//conv_kernel.stride[0]
    num_groups = conv_kernel.groups
    in_channels_per_group = c_in//num_groups
    out_channels_per_group = conv_kernel.out_channels//num_groups
    
    num_ops_out_element = 2 * conv_kernel.kernel_size[0] * in_channels_per_group
    num_ops_per_group = num_ops_out_element * length_out * out_channels_per_group * batchsize
    num_ops_total = num_ops_per_group * num_groups
    c1d_perf_kpis.num_ops = num_ops_total

    input_size_bits = DTYPE_BITS * volume(in_tensor)
    weight_size_bits = DTYPE_BITS * conv_kernel.out_channels * conv_kernel.kernel_size[0] * in_channels_per_group
    output_size_bits = DTYPE_BITS * batchsize * conv_kernel.out_channels * length_out

    c1d_perf_kpis.bytes_read = (input_size_bits+weight_size_bits)//8
    c1d_perf_kpis.bytes_written = (output_size_bits)//8
    memory_time = (c1d_perf_kpis.bytes_read + c1d_perf_kpis.bytes_written)/MEM_BW_L2_MBps

    if(MATRIX_AVAILABLE == True):
        print('TBD')
        return None
    elif(VECTOR_AVAILABLE == True):     
        #Assume parallelization is done across output channel dimension
        compute_cycles_per_out_elem = conv_kernel.kernel_size[0] * (out_channels_per_group + NUM_VECTOR_OPS_PER_CYCLE - 1)//NUM_VECTOR_OPS_PER_CYCLE
        compute_cycles_per_group = batchsize * length_out * compute_cycles_per_out_elem
        compute_cycles = compute_cycles_per_group * num_groups
        compute_time = compute_cycles/FREQ_MHz

        c1d_perf_kpis.execution_time_us = max(memory_time, compute_time)

    else:     
        print('TBD')
        return None



    return c1d_perf_kpis

#Embedding
def embedding_perf_kpis(in_tensor, embedding_layer):
    embd_perf_kpis = PerfKPIs()
    batchsize, num_tokens = in_tensor.shape
    num_embeddings = embedding_layer.num_embeddings
    embedding_dim = embedding_layer.embedding_dim

    input_size_bits = DTYPE_BITS * volume(in_tensor)
    weight_size_bits = DTYPE_BITS * num_embeddings * embedding_dim
    output_size_bits = input_size_bits * embedding_dim
    embd_perf_kpis.bytes_read = (input_size_bits+weight_size_bits)//8
    embd_perf_kpis.bytes_written = (output_size_bits)//8
    memory_time = (embd_perf_kpis.bytes_read + embd_perf_kpis.bytes_written)/MEM_BW_L2_MBps

    #Treat this as table lookup on scalar
    compute_cycles = 0
    compute_time = compute_cycles/FREQ_MHz
    embd_perf_kpis.num_ops += 0

    embd_perf_kpis.execution_time_us = max(memory_time, compute_time)

    return embd_perf_kpis
     
#Layernorm
def layernorm_perf_kpis(in_tensor):
    ln_perf_kpis = PerfKPIs()
    
    vol = volume(in_tensor)
    ln_perf_kpis.num_ops += NUM_LAYERNORM_OPS * vol

    input_size_bits = DTYPE_BITS * vol
    weight_size_bits = 0
    output_size_bits = input_size_bits 
    ln_perf_kpis.bytes_read = (input_size_bits+weight_size_bits)//8
    ln_perf_kpis.bytes_written = (output_size_bits)//8
    memory_time = (ln_perf_kpis.bytes_read + ln_perf_kpis.bytes_written)/MEM_BW_L2_MBps

    compute_cycles = (vol * NUM_LAYERNORM_OPS)//NUM_VECTOR_OPS_PER_CYCLE
    compute_time = compute_cycles/FREQ_MHz

    ln_perf_kpis.execution_time_us = max(memory_time, compute_time)
    
    return ln_perf_kpis

#Softmax
def softmax_perf_kpis(in_tensor):
    sftmx_perf_kpis = PerfKPIs()
    vol = volume(in_tensor)
    sftmx_perf_kpis.num_ops += NUM_SOFTMAX_OPS * vol

    input_size_bits = DTYPE_BITS * vol
    weight_size_bits = 0
    output_size_bits = input_size_bits 
    sftmx_perf_kpis.bytes_read = (input_size_bits+weight_size_bits)//8
    sftmx_perf_kpis.bytes_written = (output_size_bits)//8
    memory_time = (sftmx_perf_kpis.bytes_read + sftmx_perf_kpis.bytes_written)/MEM_BW_L2_MBps

    compute_cycles = (vol * NUM_SOFTMAX_OPS)//NUM_VECTOR_OPS_PER_CYCLE
    compute_time = compute_cycles/FREQ_MHz

    sftmx_perf_kpis.execution_time_us = max(memory_time, compute_time)
    
    return sftmx_perf_kpis

#Embedding
def mlp_perf_kpis(in_tensor, mlp):
    linear_0 = mlp[0]
    linear_1 = mlp[2]
    MLP_perf_kpis = PerfKPIs()

    batchsize, length_in, c_in = in_tensor.shape

    linear_0_weight_transposed = torch.transpose(linear_0.weight, 0, 1)
    linear_1_weight_transposed = torch.transpose(linear_1.weight, 0, 1)

    for count in range(batchsize):
        linear_0_perf_kpis = matmul_2d_perf_kpis(in_tensor[count], linear_0_weight_transposed)
        MLP_perf_kpis.add(linear_0_perf_kpis)

    intermediate_output = torch.zeros((length_in, linear_0.out_features))
    for count in range(batchsize):
        linear_1_perf_kpis = matmul_2d_perf_kpis(intermediate_output, linear_1_weight_transposed)
        MLP_perf_kpis.add(linear_1_perf_kpis)
    
    return MLP_perf_kpis
