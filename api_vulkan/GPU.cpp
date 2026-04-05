#include "GPU.hpp"
#include <iostream>
#include <print>
#include <format>

void VulkanBuffer::Destroy(VkDevice device) {
    if (mapped) vkUnmapMemory(device, memory);
    if (buffer) vkDestroyBuffer(device, buffer, nullptr);
    if (memory) vkFreeMemory(device, memory, nullptr);
    buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; mapped = nullptr;
}

GPU::~GPU() {
    ReleaseResources();
}

auto GPU::Init(size_t bridgeSize) -> std::expected<void, gpu_error> {
    return CreateInstance()
        .and_then([this] { return PickDevice(); })
        .and_then([this] { return CreateLogicalDevice(); })
        .and_then([this] { return CreateSyncObjects(); })
        .and_then([this, bridgeSize]() -> std::expected<void, gpu_error> {
        // 分配三塊核心 Buffer
        m_uploadHeap = CreateBuffer(bridgeSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        m_vramTemp = CreateBuffer(bridgeSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        m_readbackHeap = CreateBuffer(bridgeSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (!m_uploadHeap.buffer || !m_vramTemp.buffer || !m_readbackHeap.buffer) {
            return std::unexpected(gpu_error::ResourceCreationFailed);
        }
        return {};
            })
        .and_then([this] { return SetupLayouts(); }); // <--- 關鍵：在這裡初始化 Layout 與 DescriptorSet
}

//** --- 拆解後的實作細節(初始化) ---

auto GPU::CreateInstance() -> std::expected<void, gpu_error> {
    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instInfo, nullptr, &m_instance) != VK_SUCCESS)
        return std::unexpected(gpu_error::DeviceCreationFailed);
    return {};
}

auto GPU::PickDevice() -> std::expected<void, gpu_error> {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0) return std::unexpected(gpu_error::DeviceCreationFailed);

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());
    m_physicalDevice = devices[0]; // 預設選第一張

    // 尋找 Queue Family
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qCount, qProps.data());

    for (uint32_t i = 0; i < qCount; i++) {
        if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) m_computeIdx = i;
        if (qProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) m_copyIdx = i;
    }
    return {};
}

auto GPU::CreateLogicalDevice() -> std::expected<void, gpu_error> {
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qInfos;
    uint32_t indices[] = { m_computeIdx, m_copyIdx };
    for (uint32_t idx : indices) {
        VkDeviceQueueCreateInfo qi{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, idx, 1, &priority };
        qInfos.push_back(qi);
        if (m_computeIdx == m_copyIdx) break;
    }

    VkPhysicalDeviceSynchronization2Features sync2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, nullptr, VK_TRUE };
    VkDeviceCreateInfo dInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &sync2, 0, (uint32_t)qInfos.size(), qInfos.data() };

    if (vkCreateDevice(m_physicalDevice, &dInfo, nullptr, &m_device) != VK_SUCCESS)
        return std::unexpected(gpu_error::DeviceCreationFailed);

    vkGetDeviceQueue(m_device, m_computeIdx, 0, &m_computeQueue);
    vkGetDeviceQueue(m_device, m_copyIdx, 0, &m_copyQueue);

    // Command Pool
    VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, m_computeIdx };
    vkCreateCommandPool(m_device, &cpInfo, nullptr, &m_computePool);

    VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, m_computePool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    vkAllocateCommandBuffers(m_device, &cbAlloc, &m_computeCmd);

    return {};
}

auto GPU::CreateSyncObjects() -> std::expected<void, gpu_error> {
    VkSemaphoreTypeCreateInfo typeInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr, VK_SEMAPHORE_TYPE_TIMELINE, 0 };
    VkSemaphoreCreateInfo sInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &typeInfo };

    if (vkCreateSemaphore(m_device, &sInfo, nullptr, &m_computeSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(m_device, &sInfo, nullptr, &m_copySemaphore) != VK_SUCCESS)
        return std::unexpected(gpu_error::FenceCreationFailed);

    return {};
}

// 在 GPU.cpp 中新增一個私有函式，並在 Init 的 and_then 鏈條中呼叫它
auto GPU::SetupLayouts() -> std::expected<void, gpu_error> {
    // 1. Descriptor Set Layout (對應 register(u0))
    VkDescriptorSetLayoutBinding binding{ 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &binding };
    vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorLayout);

    // 2. Push Constant Range (256 bytes)
    VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 256 };

    // 3. Pipeline Layout
    VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &m_descriptorLayout, 1, &pushRange };
    vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_pipelineLayout);

    // 4. Descriptor Pool & Set
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &poolSize };
    vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool);

    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, m_descriptorPool, 1, &m_descriptorLayout };
    vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet);

    // 5. 更新 Descriptor Set 指向 m_vramTemp
    VkDescriptorBufferInfo bInfo{ m_vramTemp.buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bInfo, nullptr };
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    return {};
}
//**

VulkanBuffer GPU::CreateBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    VulkanBuffer vBuffer; vBuffer.size = size;
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, (VkDeviceSize)size, usage };
    vkCreateBuffer(m_device, &bufferInfo, nullptr, &vBuffer.buffer);

    VkMemoryRequirements req; vkGetBufferMemoryRequirements(m_device, vBuffer.buffer, &req);
    VkPhysicalDeviceMemoryProperties memProp; vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProp);

    uint32_t type = 0;
    for (uint32_t i = 0; i < memProp.memoryTypeCount; i++) {
        if ((req.memoryTypeBits & (1 << i)) && (memProp.memoryTypes[i].propertyFlags & props) == props) {
            type = i; break;
        }
    }

    VkMemoryAllocateInfo alloc{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, type };
    vkAllocateMemory(m_device, &alloc, nullptr, &vBuffer.memory);
    vkBindBufferMemory(m_device, vBuffer.buffer, vBuffer.memory, 0);

    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) vkMapMemory(m_device, vBuffer.memory, 0, size, 0, &vBuffer.mapped);
    return vBuffer;
}

void GPU::RecordXorShader(size_t vramOffset, size_t size, uint32_t iters) {
    if (m_computeCmd == VK_NULL_HANDLE || m_computePipeline == VK_NULL_HANDLE) return;

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(m_computeCmd, &beginInfo);

    // 1. Copy Upload -> VRAM
    VkBufferCopy region{ 0, (VkDeviceSize)vramOffset, (VkDeviceSize)size };
    vkCmdCopyBuffer(m_computeCmd, m_uploadHeap.buffer, m_vramTemp.buffer, 1, &region);

    // 2. Barrier (確保 Copy 完才算)
    VkBufferMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.buffer = m_vramTemp.buffer;
    barrier.offset = vramOffset;
    barrier.size = size;

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 1, &barrier };
    vkCmdPipelineBarrier2(m_computeCmd, &dep);

    // 3. 綁定已經建好的 Pipeline (這行取代了你原本報錯的地方)
    vkCmdBindPipeline(m_computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);

    vkCmdBindDescriptorSets(m_computeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    // 4. Push Constants
    struct { uint32_t b; uint32_t s; uint32_t i; uint32_t p; uint32_t d[60]; } cb = { (uint32_t)vramOffset, (uint32_t)size, iters, 0 };
    vkCmdPushConstants(m_computeCmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 256, &cb);

    // 5. Dispatch
    vkCmdDispatch(m_computeCmd, ((uint32_t)(size / 4) + 63) / 64, 1, 1);
}

void GPU::ResetCommandList() { vkResetCommandBuffer(m_computeCmd, 0); }

void GPU::DownloadFromVram(size_t size) {
    // 屏障：確保前面的運算/搬入已經完成
    VkBufferMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; // 保險起見，等所有指令
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;        // 準備 Copy 出去
    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barrier.buffer = m_vramTemp.buffer;
    barrier.offset = 0;
    barrier.size = (VkDeviceSize)size;

    VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depInfo.bufferMemoryBarrierCount = 1;
    depInfo.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(m_computeCmd, &depInfo);

    // 從 VramTemp 搬回 ReadbackHeap (CPU可讀區)
    VkBufferCopy downloadRegion{};
    downloadRegion.srcOffset = 0;
    downloadRegion.dstOffset = 0;
    downloadRegion.size = (VkDeviceSize)size;
    vkCmdCopyBuffer(m_computeCmd, m_vramTemp.buffer, m_readbackHeap.buffer, 1, &downloadRegion);

    // 必須在這裡結束錄製
    vkEndCommandBuffer(m_computeCmd);
}

auto GPU::BuildComputePipeline(const std::vector<uint32_t>& spirvCode) -> std::expected<void, gpu_error> {
    // 1. 建立 Shader Module
    VkShaderModule computeModule = CreateShaderModule(spirvCode);
    if (computeModule == VK_NULL_HANDLE) return std::unexpected(gpu_error::ResourceCreationFailed);

    // 2. 定義 Shader Stage (進入點通常是 "main")
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = computeModule;
    stageInfo.pName = "main"; // 必須與 HLSL 中的進入點名稱一致

    // 3. 定義 Pipeline 建立資訊 (這就是你漏掉的 pipelineInfo)
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = m_pipelineLayout; // 確保 m_pipelineLayout 已經建立
    pipelineInfo.stage = stageInfo;

    // 4. 真正向驅動程式要求建立 Pipeline
    VkResult res = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_computePipeline);

    // 建立完後 Shader Module 就可以先刪掉了（已經燒進 Pipeline 了）
    vkDestroyShaderModule(m_device, computeModule, nullptr);

    if (res != VK_SUCCESS) {
        return std::unexpected(gpu_error::ResourceCreationFailed);
    }

    return {};
}

auto GPU::CreateShaderModule(const std::vector<uint32_t>& code) -> VkShaderModule {
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, code.size() * sizeof(uint32_t), code.data() };
    VkShaderModule module;
    vkCreateShaderModule(m_device, &createInfo, nullptr, &module);
    return module;
}

uint64_t GPU::ExecuteAndSignal() {
    m_computeValue++;

    VkTimelineSemaphoreSubmitInfo timelineInfo{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &m_computeValue;

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pNext = &timelineInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_computeCmd; // 確保這裡不是空的
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_computeSemaphore;

    if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        return 0;
    }
    return m_computeValue;
}

void GPU::Wait() {
    VkSemaphoreWaitInfo wait{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1, &m_computeSemaphore, &m_computeValue };
    vkWaitSemaphores(m_device, &wait, UINT64_MAX);
}

void GPU::ReleaseResources() {
    if (!m_device) return;
    vkDeviceWaitIdle(m_device);
    if (m_computePipeline) vkDestroyPipeline(m_device, m_computePipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    if (m_descriptorPool) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    if (m_descriptorLayout) vkDestroyDescriptorSetLayout(m_device, m_descriptorLayout, nullptr);
    m_uploadHeap.Destroy(m_device);
    m_vramTemp.Destroy(m_device);
    m_readbackHeap.Destroy(m_device);
    if (m_computePool) vkDestroyCommandPool(m_device, m_computePool, nullptr);
    if (m_computeSemaphore) vkDestroySemaphore(m_device, m_computeSemaphore, nullptr);
    if (m_copySemaphore) vkDestroySemaphore(m_device, m_copySemaphore, nullptr);
    vkDestroyDevice(m_device, nullptr); m_device = VK_NULL_HANDLE;
    if (m_instance) vkDestroyInstance(m_instance, nullptr); m_instance = VK_NULL_HANDLE;
}