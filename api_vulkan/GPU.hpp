#pragma once
#include <vulkan/vulkan.h>
#include <expected>
#include <vector>

#pragma comment(lib, "vulkan-1.lib")

enum class gpu_error {
    DeviceCreationFailed = 5546,
    QueueCreationFailed = 5547,
    FenceCreationFailed = 5548,
    ResourceCreationFailed = 5551
};

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t size = 0;
    void Destroy(VkDevice device);
};

class GPU {
public:
    GPU() = default;
    ~GPU();

    auto Init(size_t bridgeSize) -> std::expected<void, gpu_error>;
    void ReleaseResources();

    // Pipeline 初始化
    auto BuildComputePipeline(const std::vector<uint32_t>& spirvCode) -> std::expected<void, gpu_error>;

    // 記憶體位址存取 (已統一，不再重複定義)
    void* GetUploadAddress()   const { return m_uploadHeap.mapped; }
    void* GetVramAddress()     const { return m_vramTemp.mapped; }
    void* GetReadbackAddress() const { return m_readbackHeap.mapped; }

    // 指令錄製與執行
    void RecordXorShader(size_t vramOffset, size_t size, uint32_t iters);
    void ResetCommandList();
    uint64_t ExecuteAndSignal();
    void Wait();
    void DownloadFromVram(size_t size);

private:
    // 內部輔助函式
    auto CreateInstance() -> std::expected<void, gpu_error>;
    auto PickDevice() -> std::expected<void, gpu_error>;
    auto CreateLogicalDevice() -> std::expected<void, gpu_error>;
    auto CreateSyncObjects() -> std::expected<void, gpu_error>;
    auto CreateShaderModule(const std::vector<uint32_t>& code) -> VkShaderModule;
    auto SetupLayouts() -> std::expected<void, gpu_error>;
    VulkanBuffer CreateBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props);

    // 物件成員
    VkInstance       m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice         m_device = VK_NULL_HANDLE;

    VkQueue          m_computeQueue = VK_NULL_HANDLE;
    VkQueue          m_copyQueue = VK_NULL_HANDLE;
    VkCommandPool    m_computePool = VK_NULL_HANDLE;
    VkCommandBuffer  m_computeCmd = VK_NULL_HANDLE;

    uint32_t         m_computeIdx = 0;
    uint32_t         m_copyIdx = 0;

    VkSemaphore      m_computeSemaphore = VK_NULL_HANDLE;
    VkSemaphore      m_copySemaphore = VK_NULL_HANDLE;
    uint64_t         m_computeValue = 0;
    uint64_t         m_copyValue = 0;

    VulkanBuffer     m_uploadHeap;
    VulkanBuffer     m_vramTemp;
    VulkanBuffer     m_readbackHeap;

    // --- Compute Pipeline 資源 ---
    VkPipeline            m_computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet       m_descriptorSet = VK_NULL_HANDLE;
};