#include "GPU.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <print>
#include <numeric>
#include <cstring>
#include <omp.h>
#include <fstream> // 必須包含此標頭來讀取 .spv

using namespace std;

// 輔助函式：讀取二進位 SPIR-V 檔案
vector<uint32_t> LoadSPIRV(const string& filename) {
    ifstream file(filename, ios::ate | ios::binary);
    if (!file.is_open()) {
        std::println("[!] 找不到 Shader 檔案: {}", filename);
        return {};
    }
    size_t fileSize = (size_t)file.tellg();
    vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();
    return buffer;
}

// CPU 高強度多核運算
void CpuXorMultiCore(uint32_t* data, size_t count, uint32_t iters) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)count; ++i) {
        uint32_t val = data[i];
        uint32_t threadID = (uint32_t)i;
        for (uint32_t j = 0; j < iters; ++j) {
            val ^= (0xABCDEFAF + j + (threadID & 0xFF));
            val = (val << 3) | (val >> (32 - 3));
        }
        data[i] = val;
    }
}

int main() {
    std::println("\n    === Vulkan 1.4 GPGPU vs CPU 多核心極限對決 V2.0 ===");

    const size_t dataSize = 256ULL * 1024 * 1024;
    const size_t elementCount = dataSize / sizeof(uint32_t);

    GPU gpu;
    // 1. 初始化資源與 Layout (SetupLayouts 會在 Init 內部被調用)
    if (!gpu.Init(dataSize + 1024 * 1024)) {
        std::println("[!] GPU 初始化失敗。");
        return -1;
    }

    // 2. 編譯與建立 Pipeline (必須在錄製指令之前完成)
    auto spirv = LoadSPIRV("SuperCompute.spv");
    if (spirv.empty()) return -1;

    if (!gpu.BuildComputePipeline(spirv)) {
        std::println("[!] Pipeline 建立失敗。");
        return -1;
    }

    vector<uint32_t> initialData(elementCount);
    std::iota(initialData.begin(), initialData.end(), 0);

    vector<uint32_t> testConfigs = { 64, 256, 1024 };

    std::println("[*] 數據規模: 256 MB ({} 個 uint32)", elementCount);
    std::println("[*] CPU 線程數: {} (OpenMP)", omp_get_max_threads());
    std::println("| 迭代次數 | CPU(多核)ms | GPU (ms) | 加速比 | 驗證 | 算力 (GOPS) |");
    std::println("|----------|-------------|----------|--------|------|-------------|");

    for (uint32_t iters : testConfigs) {
        // --- A. CPU 運算 ---
        vector<uint32_t> cpuRes = initialData;
        auto c_start = chrono::high_resolution_clock::now();
        CpuXorMultiCore(cpuRes.data(), cpuRes.size(), iters);
        auto c_end = chrono::high_resolution_clock::now();
        double cpuMs = chrono::duration<double, milli>(c_end - c_start).count();

        // --- B. GPU 運算 ---
        // 直接 memcpy 到 Upload 地址
        memcpy(gpu.GetUploadAddress(), initialData.data(), dataSize);

        gpu.ResetCommandList();
        // 錄製包含了：Copy -> Barrier -> Dispatch -> Barrier -> Copy Back
        gpu.RecordXorShader(0, dataSize, iters);
        gpu.DownloadFromVram(dataSize);

        auto g_start = chrono::high_resolution_clock::now();
        gpu.ExecuteAndSignal();
        gpu.Wait();
        auto g_end = chrono::high_resolution_clock::now();
        double gpuMs = chrono::duration<double, milli>(g_end - g_start).count();

        // --- C. 結果回讀與驗證 ---
        uint32_t* gpuResPtr = (uint32_t*)gpu.GetReadbackAddress();

        bool match = true;
        for (size_t i = 0; i < elementCount; ++i) {
            if (gpuResPtr[i] != cpuRes[i]) {
                match = false;
                std::println("\n[!] 驗證失敗於 Index[{}]: CPU=0x{:08X}, GPU=0x{:08X}",
                    i, cpuRes[i], gpuResPtr[i]);
                break;
            }
        }

        double gops = (double(elementCount) * iters) / (gpuMs * 1e6);

        std::println("| {:>8} | {:>11.1f} | {:>8.1f} | {:>6.1f}x | {} | {:>11.2f} |",
            iters, cpuMs, gpuMs, cpuMs / gpuMs, match ? "PASS" : "FAIL", gops);
    }

    std::println("------------------------------------------------------------------");
    std::println("[*] 測試結束。");
    return 0;
}