#pragma once

// 根據編譯環境自動切換 導出 (Export) 或 導入 (Import)
#ifdef GPU_EXPORTS
#define GPU_API __declspec(dllexport)
#else
#define GPU_API __declspec(dllimport)
#endif

#include <expected>
#include <vector>
#include <cstdint>

/**
 * @brief 錯誤代碼定義
 */
enum class gpu_error {
    Success = 0,
    DeviceCreationFailed = 5546,
    QueueCreationFailed = 5547,
    FenceCreationFailed = 5548,
    CanUseDeviceFailed = 5549,
    ResourceCreationFailed = 5551,
    DestroyBufferFailed = 5552,
};

enum class mod {
    GRAPHICS,
    COMPUTE,
    BOTH,
};

/**
 * @brief 封裝後的緩衝區結構
 */
struct GPU_API VulkanBuffer {
    void* internalBuffer = nullptr;     // 實際上存放 VkBuffer
    void* internalAllocation = nullptr; // 實際上存放 VmaAllocation
    size_t size = 0;
    void* mapped = nullptr;             // CPU 可直接存取的位址 (Host Visible)
};

// 前置聲明，避免引入整個 vulkan.h
struct VkShaderModule_T;
typedef struct VkShaderModule_T* VkShaderModule;
struct VkPipeline_T;
typedef struct VkPipeline_T* VkPipeline;

/**
 * @brief 通用運算配置常數 (對齊 std430 規範)
 */
struct ComputeConstants {
    uint32_t dataOffset;        // 資料在原始文件中的偏移量 (新加入)
    uint32_t currentChunkSize;  // 當前這一段分段的大小 (新加入)
    uint32_t params[2];         // 預留空間 (例如 iters, key 等)
};

template <typename T>
constexpr T AlignUp(T value, T alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief GPU 運算核心類別
 */
class GPU_API GPU {
public:
    GPU();
    ~GPU();

    // 禁止複製 (避免指標重複釋放)
    GPU(const GPU&) = delete;
    GPU& operator=(const GPU&) = delete;

    /**
     * @brief 初始化 GPU 環境與記憶體池
     * @param bridgeSize 預計分配的緩衝區大小
     * @param preferIntegrated 是否偏好使用內顯 (用於省電或特定測試)
     * @param _mod 運行模式 (影響隊列選取與 Layout 設定)
     */
    auto Init(size_t bridgeSize, bool preferIntegrated, mod _mod) -> std::expected<void, gpu_error>;

    /**
     * @brief 釋放所有 Vulkan 資源與記憶體
     */
    void ReleaseResources();

    /**
     * @brief 編譯並建立計算管線
     * @param spirvCode 著色器二進位碼 (SPIR-V)
     */
    auto BuildComputePipeline(const std::vector<uint32_t>& spirvCode) -> std::expected<void, gpu_error>;

    /**
     * @brief 取得各個緩衝區的 CPU 映射位址
     */
    void* GetUploadAddress()   const;
    void* GetVramAddress()     const;
    void* GetReadbackAddress() const;

    /**
     * @brief 錄製通用運算指令 (含資料同步與 Dispatch)
     * @param configs 運算常數 (對應 HLSL 的 push_constant)
     * @param targetPipeline 可選：傳入特定的 Pipeline 以切換演算法
     */
    void RecordCompute(const ComputeConstants& configs, VkPipeline targetPipeline = nullptr);

    /**
     * @brief 重置指令清單，準備下一次錄製
     */
    void ResetCommandList();

    /**
     * @brief 提交指令至 GPU 並返回 Timeline 序號
     * @return uint64_t 返回本次任務的 Ticket，用於 Wait()
     */
    uint64_t ExecuteAndSignal();

    /**
     * @brief 阻塞 CPU 直到 GPU 完成指定序號的運算
     * @param targetValue ExecuteAndSignal 回傳的數值
     */
    void Wait(uint64_t targetValue);

    // 釋放buffer資源(適用於執行中)
    auto SafeReleaseResources() -> std::expected<void, gpu_error>;

    auto ResizeBuffer(size_t bridgeSize) -> std::expected<void, gpu_error>;

private:
    // 內部實作專用的輔助函式
    auto CreateInstance() -> std::expected<void, gpu_error>;
    auto PickDevice(bool preferIntegrated, mod _mod) -> std::expected<void, gpu_error>;
    auto CreateLogicalDevice() -> std::expected<void, gpu_error>;
    auto CreateSyncObjects() -> std::expected<void, gpu_error>;
    auto SetupLayouts(mod _mod) -> std::expected<void, gpu_error>;
    auto CreateShaderModule(const std::vector<uint32_t>& code) -> VkShaderModule;

    auto DestroyBufferInternal(VulkanBuffer& res) -> std::expected<void,gpu_error>;

    // 封裝後的緩衝區建立邏輯
    VulkanBuffer CreateBuffer(size_t size, uint32_t usage, int vmaUsage);

private:
    struct Impl;
    Impl* pImpl;
};