import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# 1. Criar dados no CPU (NumPy)
data = np.random.randn(5).astype(np.float32)
print(f'Dados originais: {data}')

# 2. Escrever o Kernel em C++ (o que vai rodar na GPU)
mod = SourceModule('''
__global__ void double_array(float *a)
{
    int idx = threadIdx.x;
    a[idx] *= 2;
}
''')

# 3. Mapear a função e rodar
func = mod.get_function('double_array')
func(cuda.InOut(data), block=(5, 1, 1), grid=(1, 1))

# 4. Mostrar resultado
print(f'Dados após GPU (dobrados): {data}')