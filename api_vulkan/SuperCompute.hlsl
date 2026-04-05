// --- 1. 定義 Push Constant 結構 (必須與 C++ 的結構體對齊) ---
// 在 C++ 中我們定義了 4 個 uint + 60 個 uint (共 256 bytes)
struct Params
{
    uint g_baseIdx; // 數據在 Buffer 中的起始偏移 (Byte)
    uint g_totalSize; // 總數據大小 (Byte)
    uint g_iterations; // GPGPU 運算迭代次數
    uint g_unused; // 填充位 (Padding)
    
    // 剩下的 240 bytes 使用 uint4 陣列湊齊 (15 * 16 = 240)
    uint4 g_dummy[15];
};

// --- 2. 宣告 Vulkan 資源 ---
// [[vk::push_constant]] 讓這組參數直接從暫存器讀取，速度最快
[[vk::push_constant]] Params cb;

// register(u0) 對應到你 C++ 裡的 m_descriptorSet 綁定的 m_vramTemp
RWStructuredBuffer<uint> data : register(u0);

// --- 3. 運算核心 ---
// 每個 Thread Group 包含 64 個執行緒
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    
    // 計算當前執行緒對應的陣列索引 (Index)
    // 因為是 uint 陣列，所以要把 Byte 偏移除以 4
    uint iIdx = dtid.x + (cb.g_baseIdx / 4);
    
    // 邊界檢查：防止存取超過分配的 VRAM 空間
    if ((iIdx * 4) >= cb.g_totalSize)
    {
        return;
    }

    // 從 VRAM 讀取原始數據
    uint val = data[iIdx];
    
    // --- 4. GPGPU 模擬運算邏輯 ---
    // 這裡執行高強度的位元運算與循環，用來測試 GPU 吞吐量
    [loop]
    for (uint i = 0; i < cb.g_iterations; i++)
    {
        // 1. 異或運算 (XOR)
        val ^= (0xABCDEFAF + i + (iIdx & 0xFF));
        
        // 2. 循環左移 (Bit Rotation) - 這是 GPGPU 常用技巧
        val = (val << 3) | (val >> 29);
    }

    // 將運算結果寫回 VRAM
    data[iIdx] = val;
}