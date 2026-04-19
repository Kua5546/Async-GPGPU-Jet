# Vulkan-Async-GPGPU-Jet V2🚀

這是一個基於 **Vulkan** 的高效能異步計算（Async Compute）實驗專案。透過獨立的 **Compute Queue** 與 **Copy Queue** 並行運作，在處理大規模數據時達成顯著的性能提升。

## 核心技術亮點
* **異步管線重疊 (Async Overlap)**：實作 Compute 與 Copy 引擎並行，減少 GPU 空轉時間。
* **資源屏障優化 (Resource Barrier)**：精確控制 UAV Barrier，解決了大規模數據同步中的一致性風險（Stale Data Prevention）。
* **大規模數據壓測**：針對 256 MB 數據量進行循環計算驗證，確保在極端負載下的穩定性。

## 🚀 性能實測 (Benchmark)
測試環境：
* GPU : NVIDIA GeForce RTX 4060 8GB
* CPU : I5 12400F
* RAM : 24GB 2400 MT/S

| 數據規模 | 迭代次數 (Iterations) | 加速比 (vs CPU/Sync) | 數據校驗 (Validation) |
| :--- | :--- | :--- | :--- |
| 300 MB | 64 | **30.0x +** | **PASS** |
| 300 MB | 512 | **200.0x +** | **PASS** |

> **技術觀察**：隨著迭代次數增加，GPU 的吞吐量優勢（Throughput）呈線性爆發，成功抵銷了 PCIe 頻寬帶來的初始延遲。

## 🛠️ 開發環境
- **Language**: C++ 23
- **Graphics API**: Vulkan 1.4.341.1
- **Shader**: HLSL (CS_6_0)
- **IDE**: Visual Studio 2026 (.slnx ready)

## 📂 專案結構
- `API/main.cpp`: 測試邏輯與高精度計時器實作。
- `API/GPU.cpp`: Vulkan 核心架構、指令列表管理與異步同步邏輯。
- `API/SuperCompute.hlsl`: 最佳化的 GPGPU 計算核心。
- 
## 已知問題

## 暫時停止開發
* N

# V3預告(將在另一個專案的V1)
* 使用新語言重寫
> 新語言將用C做編譯器，但是還是使用Vulkan
> 目前想法設計 : 簡單的語法 、 內建Vulkan(提供簡化的語法)，類似PY但是效能要更高且使用顯式，比PY更接近底層，希望能跟C/CPP相容

## 其他&不足

## 2026 4/12提交
