#define GPU_EXPORTS           // 必須在 include GPU.hpp 之前，確保是 Export 模式
#define VMA_IMPLEMENTATION    // 僅在實作檔定義一次

// 1. 先處理第三方庫的警告屏蔽
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4251) 
#include "VulkanMemoryAllocator-3.3.0/include/vk_mem_alloc.h"
#pragma warning(pop)
#else
    // Linux/GCC 屏蔽警告的方式
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable" 
#include "VulkanMemoryAllocator-3.3.0/include/vk_mem_alloc.h"
#pragma GCC diagnostic pop
#endif

// 2. 引入自己的標頭檔
#include "GPU.hpp"

// 3. 其他標準庫
#include <print>
#include <iostream>
#include <set>      
#include <vector>   
#include <algorithm> 
#include <vulkan/vulkan.h>

#ifndef VK_API_VERSION_1_4
#define VK_API_VERSION_1_4 VK_MAKE_API_VERSION(0, 1, 4, 0)
#endif

#ifndef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES (VkStructureType)1000540000

#endif

struct GPU::Impl {
    VkInstance       m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice         m_device = VK_NULL_HANDLE;
    VmaAllocator     m_allocator = VK_NULL_HANDLE;
    VkDeviceSize     m_minAlignment = 0;

    VkQueue          m_computeQueue = VK_NULL_HANDLE;
    VkQueue          m_copyQueue = VK_NULL_HANDLE;
    VkCommandPool    m_computePool = VK_NULL_HANDLE;
    VkCommandPool    m_copyPool = VK_NULL_HANDLE;
    VkCommandBuffer  m_computeCmd = VK_NULL_HANDLE;
    VkCommandBuffer  m_copyCmd = VK_NULL_HANDLE; // 專門錄製搬運指令

    uint32_t         m_computeIdx = 0;
    uint32_t         m_graphicsIdx = 0;
    uint32_t         m_copyIdx = 0;

    VkSemaphore      m_computeSemaphore = VK_NULL_HANDLE;
    VkSemaphore      m_copySemaphore = VK_NULL_HANDLE;
    uint64_t         m_computeValue = 0;
    uint64_t         m_copyValue = 0;

    VulkanBuffer     m_uploadHeap;
    VulkanBuffer     m_vramTemp;
    VulkanBuffer     m_readbackHeap;

    VkPipeline            m_computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet       m_descriptorSet = VK_NULL_HANDLE;

    std::mutex m_submitMutex;

    bool m_isLocked = false;
};

GPU::GPU() : pImpl(new Impl()) {
    pImpl->m_isLocked = false;
}

GPU::~GPU() {
    ReleaseResources();
    delete pImpl;
}

auto GPU::Init(size_t bridgeSize, bool cpu_GPU ,mod _mod) -> std::expected<void, gpu_error> {
    return CreateInstance()
        .and_then([this,cpu_GPU,_mod] { return PickDevice(cpu_GPU, _mod); })
        .and_then([this] { return CreateLogicalDevice(); }) // 這裡會初始化 m_allocator
        .and_then([this] { return CreateSyncObjects(); })
        .and_then([this, bridgeSize]() -> std::expected<void, gpu_error> {

            // 使用 VMA 建立緩衝區
            pImpl -> m_uploadHeap = CreateBuffer(bridgeSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST);

            pImpl-> m_vramTemp = CreateBuffer(bridgeSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

            pImpl-> m_readbackHeap = CreateBuffer(bridgeSize,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST);

            if (!pImpl->m_uploadHeap.internalBuffer ||
                !pImpl->m_vramTemp.internalBuffer ||
                !pImpl->m_readbackHeap.internalBuffer) {
                return std::unexpected(gpu_error::ResourceCreationFailed);
            }
            return {};
        })
        .and_then([this, _mod] { return SetupLayouts(_mod); });
}

//** --- 拆解後的實作細節(初始化) ---

auto GPU::CreateInstance() -> std::expected<void, gpu_error> {
    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instInfo, nullptr, &pImpl -> m_instance) != VK_SUCCESS)
        return std::unexpected(gpu_error::DeviceCreationFailed);
    return {};
}

auto GPU::PickDevice(bool cpu_GPU, mod _mod) -> std::expected<void, gpu_error> {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(pImpl->m_instance, &count, nullptr);
    if (count == 0) return std::unexpected(gpu_error::DeviceCreationFailed);

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(pImpl->m_instance, &count, devices.data());

    VkPhysicalDevice chosenDevice = VK_NULL_HANDLE;
    int highestScore = -1;

    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        int score = 0;
        // 模式 A：強制選擇內顯 (測試用)
        if (cpu_GPU && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            chosenDevice = device;
            break;
        }

        // 模式 B：選擇效能最強的卡 (通常是你的 RTX 4060)
        if (!cpu_GPU) {
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            {
                // 以 maxImageDimension2D 作為權重，並對獨顯加權
                int currentScore = static_cast<int>(props.limits.maxImageDimension2D);
                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) currentScore += 10000;

                if (currentScore > highestScore) {
                    highestScore = currentScore;
                    chosenDevice = device;
                }
            }
        }
    }

    if (chosenDevice == VK_NULL_HANDLE) return std::unexpected(gpu_error::CanUseDeviceFailed);

    pImpl->m_physicalDevice = chosenDevice;

    // 取得選定設備的屬性 (包含你繞回覆蓋需要的對齊資訊)
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(chosenDevice, &deviceProperties);
    std::println("Selected GPU: {}", deviceProperties.deviceName);

    // 【關鍵】抓取硬體對齊要求：這決定了你「0~255 繞回」時的 Offset 必須是多少的倍數
    // 例如 4060 通常要求 32 或 64 bytes 對齊
    pImpl->m_minAlignment = deviceProperties.limits.minStorageBufferOffsetAlignment;

    // 尋找 Queue Family - 解決效能陷阱：尋找真正的 Async Compute 隊列
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pImpl->m_physicalDevice, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pImpl->m_physicalDevice, &qCount, qProps.data());

    pImpl->m_computeIdx = uint32_t(-1);
    pImpl->m_copyIdx = uint32_t(-1);

    switch (_mod)
    {
        case mod::COMPUTE:
            // 第一輪搜尋：找專用隊列 (Dedicated Queues)
            for (uint32_t i = 0; i < qCount; i++) {
                auto flags = qProps[i].queueFlags;

                // 找「純計算」隊列：有 Compute 但沒有 Graphics，這才是真正的 Async Compute
                if ((flags & VK_QUEUE_COMPUTE_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT)) {
                    pImpl->m_computeIdx = i;
                }
                // 找「純傳輸」隊列：這在搬運 259MB 資料時不會卡到計算
                if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
                    pImpl->m_copyIdx = i;
                }
            }
            if (pImpl->m_computeIdx == uint32_t(-1)) {
                for (uint32_t i = 0; i < qCount; i++) {
                    if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                        pImpl->m_computeIdx = i;
                        break;
                    }
                }
            }
            break;
        case mod::GRAPHICS:
            // 第二輪搜尋：如果沒找到專用的，才找通用的
            for (uint32_t i = 0; i < qCount; i++) {
                if (qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    pImpl->m_computeIdx = i; // 在圖形模式下，這個 Index 同時負責渲染
                    break;
                }
            }
            // 圖形模式也需要搬運工，找專用傳輸隊列來加速貼圖上傳
            for (uint32_t i = 0; i < qCount; i++) {
                if ((qProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                    !(qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                    pImpl->m_copyIdx = i;
                    break;
                }
            }
            break;
        case mod::BOTH:
            // MORE
            //GRAPHICS
            for (uint32_t i = 0; i < qCount; i++) {
                if (qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    pImpl->m_graphicsIdx = i;
                    break;
                }
            }
            //COMPUTE
            for (uint32_t i = 0; i < qCount; i++) {
                if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    pImpl->m_computeIdx = i;
                    break;
                }
            }

            // 3. 如果沒找到獨立計算隊列，才退而求其次用同一個
            if (pImpl->m_computeIdx == uint32_t(-1)) {
                pImpl->m_computeIdx = pImpl->m_graphicsIdx;
            }

            for (uint32_t i = 0; i < qCount; i++) {
                auto flags = qProps[i].queueFlags;
                if ((flags & VK_QUEUE_TRANSFER_BIT) &&
                    !(flags & VK_QUEUE_GRAPHICS_BIT) &&
                    !(flags & VK_QUEUE_COMPUTE_BIT)) {
                    pImpl->m_copyIdx = i;
                    break;
                }
            }

            break;
    }

    if (pImpl->m_computeIdx == uint32_t(-1)) {
        std::println("Error: No queue family supports compute operations.");
        return std::unexpected(gpu_error::QueueCreationFailed);
    }

    if (pImpl->m_copyIdx == uint32_t(-1)) {
        // 傳輸通常可以跟隨計算隊列
        pImpl->m_copyIdx = pImpl->m_computeIdx;
    }

    std::println("Compute Queue Family: {} | Transfer Queue Family: {}", pImpl->m_computeIdx, pImpl->m_copyIdx);

    return {};
}

auto GPU::CreateLogicalDevice() -> std::expected<void, gpu_error> {
    // 1. 設定 Queue 請求 - 簡化邏輯，每個家族只請 1 個
    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { pImpl->m_graphicsIdx, pImpl->m_computeIdx, pImpl->m_copyIdx };

    for (uint32_t index : uniqueQueueFamilies) {
        if (index == uint32_t(-1)) continue;
        VkDeviceQueueCreateInfo qInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qInfo.queueFamilyIndex = index;
        qInfo.queueCount = 1; // 始終請求 1 個，確保硬體相容性
        qInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(qInfo);
    }

    // 2. 啟用特性 (補上 createInfo.queueCreateInfoCount)
    VkPhysicalDeviceVulkan14Features features14 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES };
    features14.pushDescriptor = VK_TRUE;

    VkPhysicalDeviceVulkan13Features features13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features13.synchronization2 = VK_TRUE;
    features13.pNext = &features14;

    VkPhysicalDeviceVulkan12Features features12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.timelineSemaphore = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.pNext = &features13;

    VkDeviceCreateInfo createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    createInfo.pNext = &features12;
    createInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size(); // 補上這行
    createInfo.pQueueCreateInfos = queueCreateInfos.data();             // 補上這行

    if (vkCreateDevice(pImpl->m_physicalDevice, &createInfo, nullptr, &pImpl->m_device) != VK_SUCCESS) {
        return std::unexpected(gpu_error::DeviceCreationFailed);
    }

    // 4. 取得 Queue 句柄 (安全取得)
    vkGetDeviceQueue(pImpl->m_device, pImpl->m_computeIdx, 0, &pImpl->m_computeQueue);
    vkGetDeviceQueue(pImpl->m_device, pImpl->m_copyIdx, 0, &pImpl->m_copyQueue);

    // 5. 初始化 VMA Allocator (確保 API 版本正確)
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_4;
    allocatorInfo.physicalDevice = pImpl->m_physicalDevice;
    allocatorInfo.device = pImpl->m_device;
    allocatorInfo.instance = pImpl->m_instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&allocatorInfo, &pImpl->m_allocator) != VK_SUCCESS) {
        return std::unexpected(gpu_error::DeviceCreationFailed);
    }

    // 6. 建立指令池與分配 Command Buffer
    VkCommandPoolCreateInfo poolInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = pImpl->m_computeIdx;

    if (vkCreateCommandPool(pImpl->m_device, &poolInfo, nullptr, &pImpl->m_computePool) != VK_SUCCESS) {
        return std::unexpected(gpu_error::QueueCreationFailed);
    }

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = pImpl->m_computePool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    // 這裡如果不崩潰了，就代表之前的 DeviceCreateInfo 或 Queue 請求有誤
    if (vkAllocateCommandBuffers(pImpl->m_device, &allocInfo, &pImpl->m_computeCmd) != VK_SUCCESS) {
        return std::unexpected(gpu_error::QueueCreationFailed);
    }

    poolInfo.queueFamilyIndex = pImpl->m_copyIdx;
    vkCreateCommandPool(pImpl->m_device, &poolInfo, nullptr, &pImpl->m_copyPool);

    VkCommandBufferAllocateInfo copyAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    copyAlloc.commandPool = pImpl->m_copyPool;
    copyAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    copyAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(pImpl->m_device, &copyAlloc, &pImpl->m_copyCmd);

    return {};
}

auto GPU::CreateSyncObjects() -> std::expected<void, gpu_error> {
    VkSemaphoreTypeCreateInfo typeInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr, VK_SEMAPHORE_TYPE_TIMELINE, 0 };
    VkSemaphoreCreateInfo sInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &typeInfo };

    if (vkCreateSemaphore(pImpl->m_device, &sInfo, nullptr, &pImpl->m_computeSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(pImpl->m_device, &sInfo, nullptr, &pImpl->m_copySemaphore) != VK_SUCCESS)
        return std::unexpected(gpu_error::FenceCreationFailed);

    return {};
}

//** 在 GPU.cpp 中新增一個私有函式，並在 Init 的 and_then 鏈條中呼叫它
auto GPU::SetupLayouts(mod _mod) -> std::expected<void, gpu_error> {
    // 1. 定義 Binding (綁定點 0: 儲存緩衝區)
    // 根據 _mod 決定 Shader 階段，COMPUTE 模式下僅需運算階段
    VkShaderStageFlags activeStages = VK_SHADER_STAGE_COMPUTE_BIT;
    if (_mod == mod::GRAPHICS || _mod == mod::BOTH) {
        activeStages |= VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = activeStages,
        .pImmutableSamplers = nullptr
    };

    // 2. 建立 Descriptor Set Layout (關鍵：必須開啟 PUSH_DESCRIPTOR 標記)
    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, // 1.4 核心特性
        .bindingCount = 1,
        .pBindings = &binding
    };

    if (vkCreateDescriptorSetLayout(pImpl->m_device, &layoutInfo, nullptr, &pImpl->m_descriptorLayout) != VK_SUCCESS) {
        return std::unexpected(gpu_error::ResourceCreationFailed);
    }

    // 3. 定義 Push Constants (用於傳遞 offset, size, iters 等小資料)
    struct PushData {
        uint32_t offset;
        uint32_t size;
        uint32_t iters;
        uint32_t padding; // 補齊 16-byte 對齊，符合 std430 規範
    };

    VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushData)
    };

    // 4. 建立 Pipeline Layout
    VkPipelineLayoutCreateInfo plInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &pImpl->m_descriptorLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange
    };

    if (vkCreatePipelineLayout(pImpl->m_device, &plInfo, nullptr, &pImpl->m_pipelineLayout) != VK_SUCCESS) {
        return std::unexpected(gpu_error::ResourceCreationFailed);
    }

    return {};
}
//**

VulkanBuffer GPU::CreateBuffer(size_t size, uint32_t usage, int vmaUsage) {
    if (pImpl->m_allocator == VK_NULL_HANDLE) return {};

    VulkanBuffer vBuffer{};
    vBuffer.size = size;

    VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = size;
    bufferInfo.usage = (VkBufferUsageFlags)usage;
    // --- 關鍵修正區 ---
    uint32_t families[] = { pImpl->m_graphicsIdx, pImpl->m_computeIdx, pImpl->m_copyIdx };
    std::set<uint32_t> uniqueIndices;

    // 過濾掉重複的 Index 與無效值 (-1)
    for (auto idx : families) {
        if (idx != uint32_t(-1)) uniqueIndices.insert(idx);
    }

    std::vector<uint32_t> familyList(uniqueIndices.begin(), uniqueIndices.end());
    if (pImpl->m_computeIdx != pImpl->m_graphicsIdx) 
    {
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = (uint32_t)familyList.size();
        bufferInfo.pQueueFamilyIndices = familyList.data();
    }
    else
    {
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = (VmaMemoryUsage)vmaUsage;

    // --- 修改處：針對大型 Buffer 使用獨立分配與 CPU 存取設定 ---
    if (vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST) {
        allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT; // 強制獨立記憶體，減少預分配開銷
    }
    else if (vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {
        // VRAM 端的 Temp Buffer 同樣建議使用 DEDICATED，確保 Resize 時能快速回收
        allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    VmaAllocationInfo resultInfo;
    VkBuffer tempBuffer;
    VmaAllocation tempAlloc;

    VkResult res = vmaCreateBuffer(pImpl->m_allocator, &bufferInfo, &allocInfo,
        &tempBuffer, &tempAlloc, &resultInfo);

    if (res == VK_SUCCESS) {
        vBuffer.internalBuffer = (void*)tempBuffer;
        vBuffer.internalAllocation = (void*)tempAlloc;
        vBuffer.mapped = resultInfo.pMappedData;
    }

    return vBuffer;
}

void GPU::RecordCompute(const ComputeConstants& configs, _size u_size, VkPipeline targetPipeline) {
    VkPipeline pipeline = (targetPipeline != VK_NULL_HANDLE) ? targetPipeline : pImpl->m_computePipeline;
    if (pipeline == VK_NULL_HANDLE || pImpl->m_computeCmd == VK_NULL_HANDLE || pImpl->m_copyCmd == VK_NULL_HANDLE) return;

    // --- 0. 先定義共用的搬運範圍 (解決未定義問題) ---
    VkBufferCopy region{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = (VkDeviceSize)configs.currentChunkSize
    };

    // --- 1. 錄製 Copy 指令 (Host -> VRAM) ---
    VkCommandBufferBeginInfo beginCopy{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(pImpl->m_copyCmd, &beginCopy);

    vkCmdCopyBuffer(pImpl->m_copyCmd, (VkBuffer)pImpl->m_uploadHeap.internalBuffer, (VkBuffer)pImpl->m_vramTemp.internalBuffer, 1, &region);

    vkEndCommandBuffer(pImpl->m_copyCmd);

    // --- 2. 錄製 Compute 指令 ---
    VkCommandBufferBeginInfo beginCompute{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(pImpl->m_computeCmd, &beginCompute);

    // Barrier 1: 確保 Copy 隊列寫入完成，Compute 才能讀取
    VkBufferMemoryBarrier2 barrier1{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, // 來源是搬運
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, // 目的是運算
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .buffer = (VkBuffer)pImpl->m_vramTemp.internalBuffer,
        .offset = 0,
        .size = region.size
    };
    VkDependencyInfo dep1{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier1 };
    vkCmdPipelineBarrier2(pImpl->m_computeCmd, &dep1);

    // 綁定與執行
    vkCmdBindPipeline(pImpl->m_computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    VkDescriptorBufferInfo bInfo{ (VkBuffer)pImpl->m_vramTemp.internalBuffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bInfo
    };
    vkCmdPushDescriptorSet(pImpl->m_computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pImpl->m_pipelineLayout, 0, 1, &write);
    vkCmdPushConstants(pImpl->m_computeCmd, pImpl->m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeConstants), &configs);

    uint32_t threadCount = AlignUp((uint32_t)configs.currentChunkSize, 4u) / 4u;
    uint32_t groupCount = AlignUp(threadCount, (uint32_t)u_size) / (uint32_t)u_size;
    vkCmdDispatch(pImpl->m_computeCmd, groupCount, 1, 1);

    // Barrier 2: 確保運算寫入完成，才能進行下載搬運
    VkBufferMemoryBarrier2 barrier2 = barrier1;
    barrier2.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; // 來源變回運算
    barrier2.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier2.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;          // 目的是搬運
    barrier2.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

    VkDependencyInfo dep2{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier2 };
    vkCmdPipelineBarrier2(pImpl->m_computeCmd, &dep2);

    // --- 3. 下載：VRAM -> Readback ---
    vkCmdCopyBuffer(pImpl->m_computeCmd, (VkBuffer)pImpl->m_vramTemp.internalBuffer, (VkBuffer)pImpl->m_readbackHeap.internalBuffer, 1, &region);

    vkEndCommandBuffer(pImpl->m_computeCmd);
}

void GPU::ResetCommandList() { vkResetCommandBuffer(pImpl->m_computeCmd, 0); }

auto GPU::BuildComputePipeline(const std::vector<uint32_t>& spirvCode) -> std::expected<void, gpu_error> {
    if (spirvCode.empty()) return std::unexpected(gpu_error::ResourceCreationFailed);

    // 1. 如果之前已經有 Pipeline，先銷毀它，防止記憶體洩漏 (Resize 或換算法時會用到)
    if (pImpl->m_computePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(pImpl->m_device, pImpl->m_computePipeline, nullptr);
        pImpl->m_computePipeline = VK_NULL_HANDLE;
    }

    VkShaderModule computeModule = CreateShaderModule(spirvCode);
    if (computeModule == VK_NULL_HANDLE) return std::unexpected(gpu_error::ResourceCreationFailed);

    VkPipelineShaderStageCreateInfo stageInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = computeModule;
    stageInfo.pName = "main"; // 提醒使用者：Shader 的進入點必須叫 main

    VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineInfo.layout = pImpl->m_pipelineLayout;
    pipelineInfo.stage = stageInfo;

    VkResult res = vkCreateComputePipelines(pImpl->m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pImpl->m_computePipeline);

    // Module 用完立刻銷毀，不影響 Pipeline 運行
    vkDestroyShaderModule(pImpl->m_device, computeModule, nullptr);

    if (res != VK_SUCCESS) return std::unexpected(gpu_error::ResourceCreationFailed);

    std::println("Compute Pipeline 建立成功！");
    return {};
}

auto GPU::CreateShaderModule(const std::vector<uint32_t>& code) -> VkShaderModule {
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, code.size() * sizeof(uint32_t), code.data() };
    VkShaderModule module;
    vkCreateShaderModule(pImpl->m_device, &createInfo, nullptr, &module);
    return module;
}

uint64_t GPU::ExecuteAndSignal() {
    std::lock_guard<std::mutex> lock(pImpl->m_submitMutex);

    pImpl->m_copyValue++;
    pImpl->m_computeValue++;

    // 1. 提交到 Copy Queue (搬運)
    VkTimelineSemaphoreSubmitInfo copyTimeline{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, nullptr, 0, nullptr, 1, &pImpl->m_copyValue };
    VkSubmitInfo copySubmit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &copyTimeline,
        .commandBufferCount = 1,
        .pCommandBuffers = &pImpl->m_copyCmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &pImpl->m_copySemaphore
    };
    vkQueueSubmit(pImpl->m_copyQueue, 1, &copySubmit, VK_NULL_HANDLE);

    // 2. 提交到 Compute Queue (運算)
    // 關鍵：這步會讓 GPU 在硬體上等待 Copy 完成才開始計算，不卡 CPU
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkTimelineSemaphoreSubmitInfo computeTimeline{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, nullptr,
        1, &pImpl->m_copyValue,    // Wait: 等待搬運完成的數值
        1, &pImpl->m_computeValue // Signal: 完成運算的數值
    };

    VkSubmitInfo computeSubmit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &computeTimeline,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &pImpl->m_copySemaphore,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &pImpl->m_computeCmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &pImpl->m_computeSemaphore
    };
    vkQueueSubmit(pImpl->m_computeQueue, 1, &computeSubmit, VK_NULL_HANDLE);

    return pImpl->m_computeValue;
}

void GPU::Wait(uint64_t targetValue) {
    VkSemaphoreWaitInfo wait{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1, &pImpl->m_computeSemaphore, &targetValue };
    vkWaitSemaphores(pImpl->m_device, &wait, UINT64_MAX);
}

void GPU::ReleaseResources() {
    if (pImpl->m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(pImpl->m_device);
    }

    auto DestroyVmaBuffer = [this](VulkanBuffer& vBuffer) {
        if (vBuffer.internalBuffer != nullptr && pImpl->m_allocator != VK_NULL_HANDLE) {
            vmaDestroyBuffer(pImpl->m_allocator, (VkBuffer)vBuffer.internalBuffer, (VmaAllocation)vBuffer.internalAllocation);
            vBuffer.internalBuffer = nullptr;
            vBuffer.internalAllocation = nullptr;
            vBuffer.mapped = nullptr;
        }
        };

    DestroyVmaBuffer(pImpl->m_uploadHeap);
    DestroyVmaBuffer(pImpl->m_vramTemp);
    DestroyVmaBuffer(pImpl->m_readbackHeap);

    if (pImpl->m_computePipeline) vkDestroyPipeline(pImpl->m_device, pImpl->m_computePipeline, nullptr);
    if (pImpl->m_pipelineLayout) vkDestroyPipelineLayout(pImpl->m_device, pImpl->m_pipelineLayout, nullptr);
    if (pImpl->m_descriptorPool) vkDestroyDescriptorPool(pImpl->m_device, pImpl->m_descriptorPool, nullptr);
    if (pImpl->m_descriptorLayout) vkDestroyDescriptorSetLayout(pImpl->m_device, pImpl->m_descriptorLayout, nullptr);
    if (pImpl->m_computeSemaphore) vkDestroySemaphore(pImpl->m_device, pImpl->m_computeSemaphore, nullptr);
    if (pImpl->m_computePool) vkDestroyCommandPool(pImpl->m_device, pImpl->m_computePool, nullptr);

    if (pImpl->m_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(pImpl->m_allocator);
        pImpl->m_allocator = VK_NULL_HANDLE;
    }

    if (pImpl->m_device) vkDestroyDevice(pImpl->m_device, nullptr);
    if (pImpl->m_instance) vkDestroyInstance(pImpl->m_instance, nullptr);
}

void* GPU::GetUploadAddress()   const {
    if (pImpl->m_isLocked) {
        std::println(stderr, "[!] 警告：資源目前處於鎖定狀態，禁止存取 Upload Heap！");
        return nullptr;
    }
    return pImpl->m_uploadHeap.mapped;
}
void* GPU::GetVramAddress()     const { return pImpl->m_vramTemp.mapped; }
void* GPU::GetReadbackAddress() const { return pImpl->m_readbackHeap.mapped; }

/**
* ResizeBuffer
**/
auto GPU::DestroyBufferInternal(VulkanBuffer& res) -> std::expected<void, gpu_error> {
    vmaDestroyBuffer(
        pImpl->m_allocator,
        (VkBuffer)res.internalBuffer,
        (VmaAllocation)res.internalAllocation // 強制轉回 VmaAllocation
    );
    res.internalAllocation = nullptr;
    res.internalBuffer = nullptr;
    res.mapped = nullptr;
    res.size = 0;

    return {};
}
auto GPU::SafeReleaseResources() -> std::expected<void,gpu_error>{
    // 禁止資源傳入
    pImpl->m_isLocked = true;

    bool hasError = false;

    //等待資源提交完成
    if (pImpl->m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(pImpl->m_device);
    }

    //釋放資源
    if (pImpl->m_allocator != VK_NULL_HANDLE) {
        // 釋放 Upload Heap
        auto a = DestroyBufferInternal(pImpl->m_uploadHeap);
        if (!a.has_value()) {
            std::cout << "Upload Heap Destroy Error" << std::endl;
            hasError = true;
        }
        // 釋放 VRAM Temp
        auto b = DestroyBufferInternal(pImpl->m_vramTemp);
        if (!b.has_value()) {
            std::cout << "VRAM Temp Destroy Error" << std::endl;
            hasError = true;
        }
        // 釋放 Readback Heap
        auto c = DestroyBufferInternal(pImpl->m_readbackHeap);
        if (!c.has_value()) {
            std::cout << "Readback Heap Destroy Error" << std::endl;
            hasError = true;
        }
    }
    
    pImpl->m_isLocked = false;
    if (hasError) {
        return std::unexpected(gpu_error::ResourceCreationFailed);
    }

    return {};
}
auto GPU::ResizeBuffer(size_t bridgeSize) -> std::expected<void, gpu_error> {
    auto releaseResult = SafeReleaseResources();
    if (!releaseResult.has_value()) {
        return std::unexpected(releaseResult.error());
    }
    if (pImpl->m_allocator != VK_NULL_HANDLE)
    {
        // 使用 VMA 建立緩衝區
        pImpl->m_uploadHeap = CreateBuffer(bridgeSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST);

        pImpl->m_vramTemp = CreateBuffer(bridgeSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        pImpl->m_readbackHeap = CreateBuffer(bridgeSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST);

        if (!pImpl->m_uploadHeap.internalBuffer ||
            !pImpl->m_vramTemp.internalBuffer ||
            !pImpl->m_readbackHeap.internalBuffer) 
        {
            return std::unexpected(gpu_error::ResourceCreationFailed);
        }
    }

    return {};
}

