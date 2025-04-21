import random
import logging
import sympy


# 生成随机多项式
def generate_random_polynomial(N, q):
    return [random.randint(0, q - 1) for _ in range(N)]


def reverse_bits(x, N):
    """
    反转一个整数的二进制位 (N 位宽)。
    :param x: 待反转的整数
    :param N: 二进制位数
    :return: 反转后的整数
    """
    result = 0
    for i in range(N):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


def generate_reversed_bit_sequence(N):
    """
    生成从 0 到 2^N-1 的数字，按 N 位倒序排列。
    :param N: 比特位数
    :return: 位反转顺序的数字列表
    """
    max_num = 2 ** N
    reversed_sequence = [reverse_bits(i, N) for i in range(max_num)]
    return reversed_sequence


def generate_psi(primitive_root, n, q):
    """
    生成旋转因子数组（psi），并按 log(n) 位倒序排列。
    :param primitive_root: 模 q 下的原始根
    :param n: NTT 长度（必须是 2 的幂）
    :param q: 模数
    :return: 按位倒序排列的旋转因子数组
    """
    # 计算 log2(n)
    log_n = n.bit_length() - 1

    # 初始化旋转因子数组
    psi = [1] * n
    current_factor = primitive_root % q

    # 逐步累乘生成旋转因子
    for i in range(1, n):
        psi[i] = (psi[i - 1] * current_factor) % q

    # 生成位倒序排列的索引
    reversed_indices = generate_reversed_bit_sequence(log_n)

    # 重新排列 psi 数组
    psi_reversed = [psi[reversed_indices[i]] for i in range(n)]

    return psi_reversed


# # 示例测试
# if __name__ == "__main__":
#     q = 17            # 模数（需要为质数）
#     n = 16             # NTT 的长度（必须是 2 的幂）
#     primitive_root = 3  # 假设 3 是模 q 下的原始根
#
#     # 生成旋转因子数组
#     psi = generate_psi(primitive_root, n, q)
#     print("位倒序后的旋转因子数组 (psi):", psi)

def ntt_forward(P, n, q, psi):
    """
    Cooley-Tukey Forward NTT.

    Parameters:
    P   -- 输入数组，长度为 n
    n   -- NTT 的大小（必须是 2 的幂）
    q   -- 模数
    psi -- 旋转因子数组（已按位倒序存储 n 个原始旋转因子的次方）

    返回:
    a   -- 经过 NTT 转换后的数组
    """
    m = 1
    k = n // 2
    a = P.copy()
    # 迭代处理每一级蝶形操作
    while m < n:
        for i in range(m):
            jFirst = 2 * i * k
            jLast = jFirst + k - 1
            psi_i = psi[m + i]

            for j in range(jFirst, jLast + 1):
                l = j + k
                t = a[j]
                u = a[l] * psi_i % q

                a[j] = (t + u) % q
                a[l] = (t - u) % q

        m *= 2
        k //= 2

    return a


def reverse_bits(x, num_bits):
    """
    将整数 x 按 num_bits 位进行倒序。
    :param x: 输入整数
    :param num_bits: 二进制位宽
    :return: 倒序后的整数
    """
    result = 0
    for i in range(num_bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


def generate_gtf(phi, N, q):
    """
    生成全局旋转因子 GTF。
    :param phi: 初始旋转因子
    :param N: NTT 大小
    :param q: 模数
    :return: GTF 数组
    """
    logN = N.bit_length() - 1
    GTF = [1] * (2 * logN)  # 初始化 GTF，大小为 2 * logN，默认值为 1
    GTF[0] = phi % q
    for i in range(1, logN):
        GTF[i] = pow(GTF[i - 1], 2, q)
    # 保持 logN 到 2*logN 的值为 1，已在初始化时完成
    return GTF


def butterfly_computation(x, y, twiddle, q):
    """
    执行 NTT 蝶形运算。
    :param x: 输入多项式系数 x
    :param y: 输入多项式系数 y
    :param twiddle: 旋转因子
    :param q: 模数
    :return: 更新后的 x 和 y
    """
    t = (y * twiddle) % q
    y_new = (x - t) % q
    x_new = (x + t) % q
    return x_new, y_new


import numpy as np

def compute_stf(n, N, GTF, q, i):
    """
    计算 STF 向量。
    :param n: 当前规模的 NTT 大小
    :param N: 总规模的 NTT 大小
    :param GTF: Twiddle factors 数组
    :param q: 模数
    :return: 计算后的 STF 数组
    """
    STF = [GTF[int(np.log2(N)) - 1 - i] for _ in range(n)]  # 初始化 STF 向量
    logN = int(np.log2(N))  # 计算 log2(N)
    logn = int(np.log2(n))  # 计算 log2(n)

    temp = logn - (logN - 1 - i)  # 计算 temp
    for j in range(temp):
        # 批量更新 STF 向量段
        step_size = 2 ** (j + logN - i - 1)
        start = step_size
        end = start + step_size
        # 段更新
        STF[start:end] = [(val * GTF[logN - 1 - j]) % q for val in STF[start:end]]
        # 段复制
        for sub_index in range(1, n // (2 ** (j + logN - i))):
            start_copy = sub_index * step_size * 2
            end_copy = start_copy + step_size * 2
            STF[start_copy:end_copy] = STF[:end]
    return STF



def optimized_ntt(P, phi, n, N, q):
    """
    优化 NTT with Run-Time Twiddle Factor Generation
    :param P: 输入多项式
    :param phi: 原始旋转因子
    :param n: BFU 数量 (2^k)
    :param N: 输入多项式的长度 (NTT 的大小)
    :param q: 模数
    :return: 变换后的多项式
    """
    logN = N.bit_length() - 1
    logn = n.bit_length() - 1

    # 1. 初始化 GTF (Global Twiddle Factors)
    GTF = generate_gtf(phi, N, q)

    # 2. NTT 主循环
    for i in range(logN):
        # 2.1 生成 STF (Stage Twiddle Factors) 对应当前阶段的旋转因子
        STF = compute_stf(n, N, GTF, q, i)

        # 2.2 执行每一阶段的 NTT
        # for j in range(2 ** i):
        #     for k in range(N // 2 // (2 ** i)):]
        temp = min((2 ** i), N // 2 // n)
        for j in range(temp):
            for k in range(N // 2 // temp):
                index = reverse_bits(j, logN - 1) + k
                index0 = (index // (N // (2 ** (i + 1)))) * (N // (2 ** i)) + index % (N // (2 ** (i + 1)))
                index1 = index0 + N // (2 ** (i + 1))

                # 蝶形运算
                P[index0], P[index1] = butterfly_computation(P[index0], P[index1], STF[k % n], q)

            # 更新 STF
            STF = [(stf * GTF[logN - i]) % q for stf in STF]

    return P


def is_prime(n):
    return sympy.isprime(n)


def find_primitive_root(q):
    return sympy.primitive_root(q)


# 进行全面测试
max_random = 20000000
for N in [2 ** i for i in range(1, 17)]:
    for n in [2 ** j for j in range(0, (N // 2).bit_length())]:
        try:
            # 随机生成模数 q（范围在 1000 万到 2000 万之间）
            q = random.randint(max_random / 2, max_random)
            while not is_prime(q):
                q = random.randint(max_random / 2, max_random)
            # 随机生成 primitive_root
            primitive_root = q - random.randint(10000, 20000)
            # 生成多项式 P
            P = generate_random_polynomial(N, q)
            # 生成旋转因子 psi
            psi = generate_psi(primitive_root, N, q)
            # 前向 NTT
            result = ntt_forward(P, N, q, psi)
            # 优化的 NTT
            transformed_P = optimized_ntt(P.copy(), primitive_root, n, N, q)
            # 检查是否匹配
            if transformed_P == result:
                print(f"q = {q}, phi = {primitive_root}, N = {N}, n = {n}: Match!")
            else:
                logging.error(f"q = {q}, N = {N}, n = {n}: Not Match!")
        except Exception as e:
            logging.error(f"Error encountered for N = {N}, n = {n}: {e}")

# 示例测试
# if __name__ == "__main__":
# # 参数定义
# q = 17334667 # 模数（需要为质数）
# N = 65536  # NTT 大小 (必须是 2 的幂)
# n = 1024  # BFU 数量 (必须是 2 的幂)
# # 生成随机多项式 P
# P = generate_random_polynomial(N, q)
# # P = [1, 1, 1, 1, 1, 1, 1, 1]  # 输入多项式
# primitive_root = 3627374  # 假设 3 是模 q 下的原始根
# # 旋转因子 ψ 的数组（假设已按位倒序顺序生成）
# psi = generate_psi(primitive_root, N, q)  # 示例数据，需根据实际生成
# # print("输入多项式 P:", P)
# # print("输入n powers of primitive_root :", psi)
# # 计算前向 NTT
# result = ntt_forward(P, N, q, psi)
# # print("NTT 结果:", result)
# # 进行优化的 NTT 变换
# transformed_P = optimized_ntt(P, primitive_root, n, N, q)
# # print("Optimized NTT with Run-time TF generator 变换后的多项式:", transformed_P)
# if transformed_P == result:
#     print(f"q = {q}, phi = {primitive_root}, N = {N}, n = {n}  Match!")
# else:
#     logging.error("Not Match!")


# 测试示例
# if __name__ == "__main__":
#     # 示例参数
#     N = 4  # NTT 大小 (必须是 2 的幂)
#     n = 2  # BFU 数量 (必须是 2 的幂)
#     q = 17  # 模数 (需要是质数)
#     phi = 3  # 初始旋转因子
#     P = [3, 5, 8, 9]  # 输入多项式
#
#     print("输入多项式 P:", P)
#
#     # 进行优化的 NTT 变换
#     transformed_P = optimized_ntt(P, phi, n, N, q)
#     print("Optimized NTT with Run-time TF generator 变换后的多项式:", transformed_P)
