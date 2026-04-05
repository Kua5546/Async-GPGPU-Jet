# Vulkan-Async-GPGPU-Jet Demo V4🚀

這是一個基於 **Vulkan** 的高效能異步計算（Async Compute）實驗專案。透過獨立的 **Compute Queue** 與 **Copy Queue** 並行運作，在處理大規模數據時達成顯著的性能提升。

## 核心技術亮點
* **異步管線重疊 (Async Overlap)**：實作 Compute 與 Copy 引擎並行，減少 GPU 空轉時間。
* **資源屏障優化 (Resource Barrier)**：精確控制 UAV Barrier，解決了大規模數據同步中的一致性風險（Stale Data Prevention）。
* **大規模數據壓測**：針對 5.6 GB 數據量進行循環計算驗證，確保在極端負載下的穩定性。

## 🚀 性能實測 (Benchmark)
測試環境：
* GPU : NVIDIA GeForce RTX 4060 8GB
* CPU : I5 12400F
* RAM : 24GB 2400 MT/S

| 數據規模 | 迭代次數 (Iterations) | 加速比 (vs CPU/Sync) | 數據校驗 (Validation) |
| :--- | :--- | :--- | :--- |
| 256 MB | 64 | 43.6x | **PASS** |
| 256 MB | 256 | **161.3x** | **PASS** |
| 256 MB | 1024 | **161.3x** | **PASS** |

> **技術觀察**：隨著迭代次數增加，GPU 的吞吐量優勢（Throughput）呈線性爆發，成功抵銷了 PCIe 頻寬帶來的初始延遲。

## 🛠️ 開發環境
- **Language**: C++ 23
- **Graphics API**: Vulkan
- **Shader**: HLSL (CS_6_0)
- **IDE**: Visual Studio 2026 (.slnx ready)

## 📂 專案結構
- `API/main.cpp`: 測試邏輯與高精度計時器實作。
- `API/GPU.cpp`: Vulkan 核心架構、指令列表管理與異步同步邏輯。
- `API/SuperCompute.hlsl`: 最佳化的 GPGPU 計算核心。
- 
## 已知問題
* 記憶體、VRAM使用量較大(日後版本會做新增)
* 尚未整理成可直接調用的Lib(日後版本會做新增)

## 暫時停止開發
* N

## 更改內容
* 更改結構(因應以後的包裝)
* 降低VRAM負擔

## 其他&不足
* 功能單一還沒法用在大部分運算
* Demo版本是AI生成的雛形日後會針對此版本錯修改和新增


## 2026 4/5提交
