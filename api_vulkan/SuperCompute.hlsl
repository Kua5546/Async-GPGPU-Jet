// 1. 定義 Push Constant 結構
struct ComputeConstants
{
    uint dataOffset; // 位元組偏移
    uint currentChunkSize; // 當前分段大小 (bytes)
    uint iters; // 迭代次數
    uint key; // 混淆密鑰 (0xABCDEFAF)
};

// 2. 宣告 Push Constant 變數 (必須指定一個名稱，例如 config)
[[vk::push_constant]] ComputeConstants config;

// 3. 宣告資料緩衝區 ( register u0 對應你的 Descriptor Set )
// 如果你使用 Descriptor Set 0, Binding 0，這裡應該是 u0
RWStructuredBuffer<uint> DataBuffer : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // 1. 嚴格檢查邊界 (以 byte 為單位比較)
    if (DTid.x * 4 >= config.currentChunkSize)
    {
        return;
    }

    // 2. 讀取資料：永遠從當前分段的第 0 個開始讀
    // 所以直接用 DTid.x，不需加 offset
    uint val = DataBuffer[DTid.x];

    // 3. 計算用於 XOR 的全局 ID (這才需要加 offset)
    uint actualThreadID = (config.dataOffset / 4) + DTid.x;

    // 4. 運算邏輯
    for (uint i = 0; i < config.iters; i++)
    {
        val ^= (config.key + i + (actualThreadID & 0xFF));
        val = (val << 3) | (val >> 29);
    }

    // 5. 寫回
    DataBuffer[DTid.x] = val;
}