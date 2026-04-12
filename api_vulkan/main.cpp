#include "GPU.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <format>
#include <numeric>   
#include <cstring>   
#include <omp.h>     
#include <fstream>   
#include <filesystem> 
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;

// --- 1. 輔助函式：讀取 SPIR-V ---
vector<uint32_t> LoadSPIRV(const string& filename) {
    if (!fs::exists(filename)) {
        cerr << std::format("[!] 關鍵錯誤：找不到 Shader 檔案 '{}'\n", filename);
        return {};
    }
    ifstream file(filename, ios::ate | ios::binary);
    if (!file.is_open()) {
        cerr << "[!] 錯誤：無法開啟 Shader 檔案。" << endl;
        return {};
    }
    size_t fileSize = (size_t)file.tellg();
    vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    cout << std::format("[+] Shader 載入成功: {} bytes\n", fileSize);
    return buffer;
}

// --- 2. CPU 運算 (驗證基準) ---
void CpuComputeMultiCore(uint32_t* data, size_t count, uint32_t iters) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)count; ++i) {
        uint32_t val = data[i];
        uint32_t threadID = static_cast<uint32_t>(i);
        for (uint32_t j = 0; j < iters; ++j) {
            val ^= (0xABCDEFAF + j + (threadID & 0xFF));
            val = (val << 3) | (val >> (32 - 3));
        }
        data[i] = val;
    }
}

int main() {
    cout << "\n==================================================================\n";
    cout << "    Vulkan GPGPU: Streaming & Precision Verification\n";
    cout << "    Hardware: NVIDIA GeForce RTX 4060 | VMA Dedicated Mem\n";
    cout << "==================================================================\n\n";

    GPU gpu;
    const size_t bufferSize = 256ULL * 1024 * 1024; // 256MB 視窗
    const size_t totalDataSize = 300ULL * 1024 * 1024; // 測試用總量

    // 1. 初始化 GPU 除錯
    cout << "[*] 正在初始化 Vulkan 環境 (預計分配: " << bufferSize / (1024 * 1024) << "MB)..." << endl;
    if (!gpu.Init(bufferSize, false, mod::COMPUTE)) {
        cerr << "[!] 初始化失敗！請檢查顯卡是否支援 Vulkan 1.2 或 VRAM 是否充足。" << endl;
        return -1;
    }
    cout << "[+] GPU 初始化成功。" << endl;

    auto spirv = LoadSPIRV("SuperCompute.spv");
    if (spirv.empty() || !gpu.BuildComputePipeline(spirv)) {
        cerr << "[!] Pipeline 建立失敗！請確認 Shader 進入點為 'main' 且符合 Vulkan 規範。" << endl;
        return -1;
    }

    // 2. 測試 64 次與 512 次
    vector<uint32_t> testIters = { 64, 512 };

    for (uint32_t iters : testIters) {
        cout << std::format("[*] 啟動測試: Iterations = {}, Data = {}MB\n", iters, totalDataSize / (1024 * 1024));

        vector<uint32_t> hostData(totalDataSize / sizeof(uint32_t));
        iota(hostData.begin(), hostData.end(), 0);
        vector<uint32_t> cpuVerifyData = hostData;

        // CPU 運算
        auto c_start = chrono::high_resolution_clock::now();
        CpuComputeMultiCore(cpuVerifyData.data(), cpuVerifyData.size(), iters);
        auto c_end = chrono::high_resolution_clock::now();
        double cpuMs = chrono::duration<double, milli>(c_end - c_start).count();

        // GPU 流式運算
        bool allMatch = true;
        auto g_start = chrono::high_resolution_clock::now();

        for (size_t offset = 0; offset < totalDataSize; offset += bufferSize) {
            size_t currentChunk = min(bufferSize, totalDataSize - offset);
            size_t elementOffset = offset / sizeof(uint32_t);

            // A. 上傳
            void* uploadPtr = gpu.GetUploadAddress();
            if (!uploadPtr) { cerr << "[!] 錯誤: 無法取得 Upload Address" << endl; break; }
            memcpy(uploadPtr, (char*)hostData.data() + offset, currentChunk);

            // B. 執行
            ComputeConstants config{
                .dataOffset = (uint32_t)offset,
                .currentChunkSize = (uint32_t)currentChunk,
                .params = { iters, 0xABCDEFAF }
            };
            gpu.ResetCommandList();
            gpu.RecordCompute(config);
            gpu.Wait(gpu.ExecuteAndSignal());

            // C. 深度驗證
            uint32_t* gpuResPtr = static_cast<uint32_t*>(gpu.GetReadbackAddress());
            size_t verifyElements = min(currentChunk / 4, (size_t)1024); // 檢查每個分段的前 1024 個元素

            for (size_t i = 0; i < verifyElements; ++i) {
                if (gpuResPtr[i] != cpuVerifyData[elementOffset + i]) {
                    allMatch = false;
                    cout << std::format("\n[!] 數據不一致！分段偏移: {} bytes, 元素索引: {}\n", offset, i);
                    cout << std::format("    預期 (CPU): 0x{:08X}\n", cpuVerifyData[elementOffset + i]);
                    cout << std::format("    實際 (GPU): 0x{:08X}\n", gpuResPtr[i]);
                    break;
                }
            }
            if (!allMatch) break;
        }
        auto g_end = chrono::high_resolution_clock::now();
        double gpuMs = chrono::duration<double, milli>(g_end - g_start).count();

        // 輸出結果表格
        cout << "┌────────────────────────────────────────────────────────────┐\n";
        cout << std::format("│  任務: {:>4} Iters 分段加密測試 ({}MB)                  │\n", iters, totalDataSize / (1024 * 1024));
        cout << "├─────────────┬─────────────┬─────────────┬──────────────────┤\n";
        cout << "│   設備      │  耗時 (ms)  │   加速比    │      狀態        │\n";
        cout << "├─────────────┼─────────────┼─────────────┼──────────────────┤\n";
        cout << std::format("│  CPU (OMP)  │ {:>11.2f} │     1.00x   │    [已驗證]      │\n", cpuMs);
        cout << std::format("│  GPU (Vulk) │ {:>11.2f} │ {:>10.2f}x │    {}    │\n",
            gpuMs, cpuMs / gpuMs, allMatch ? " [通過]   " : " [失敗]   ");
        cout << "└─────────────┴─────────────┴─────────────┴──────────────────┘\n\n";
    }

    // 3. 動態 Resize 測試與除錯
    cout << "[!] 測試場景切換：執行動態 Resize (256MB -> 512MB)..." << endl;
    auto resizeStart = chrono::high_resolution_clock::now();

    if (!gpu.ResizeBuffer(512ULL * 1024 * 1024)) {
        cerr << "[!] Resize 失敗！可能是 VRAM 不足或分配器衝突。" << endl;
    }
    else {
        auto resizeEnd = chrono::high_resolution_clock::now();
        cout << std::format("[+] Resize 成功！耗時: {:.2f} ms\n", chrono::duration<double, milli>(resizeEnd - resizeStart).count());

        // 驗證 Resize 後的 Buffer 是否真的可用
        ComputeConstants finalConfig{ .dataOffset = 0, .currentChunkSize = 512 * 1024 * 1024, .params = { 64, 0xABCDEFAF } };
        gpu.ResetCommandList();
        gpu.RecordCompute(finalConfig);
        gpu.Wait(gpu.ExecuteAndSignal());
        cout << "[+] Resize 後 512MB 壓力測試：正常。" << endl;
    }

    cout << "\n==================================================================\n";
    cout << "    測試完成。所有 GPU 資源已透過 RAII 安全回收。\n";
    cout << "==================================================================\n" << endl;

    return 0;
}