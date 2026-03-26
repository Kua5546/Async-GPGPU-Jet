# DX12-Async-GPGPU-Jet 🚀

這是一個基於 **DirectX 12** 的高效能異步計算（Async Compute）實驗專案。透過獨立的 **Compute Queue** 與 **Copy Queue** 並行運作，在處理大規模數據時達成顯著的性能提升。

## 核心技術亮點
* **異步管線重疊 (Async Overlap)**：實作 Compute 與 Copy 引擎並行，減少 GPU 空轉時間。
* **資源屏障優化 (Resource Barrier)**：精確控制 UAV Barrier，解決了大規模數據同步中的一致性風險（Stale Data Prevention）。
* **大規模數據壓測**：針對 5.6 GB 數據量進行循環計算驗證，確保在極端負載下的穩定性。

## 🚀 性能實測 (Benchmark)
測試環境：
GPU : NVIDIA GeForce RTX 4060
CPU : I5 12400F

| 數據規模 | 迭代次數 (Iterations) | 加速比 (vs CPU/Sync) | 數據校驗 (Validation) |
| :--- | :--- | :--- | :--- |
| 5.6 GB | 1 | 1.76x | **MATCH** |
| 5.6 GB | 512 | **136.5x** | **MATCH** |

> **技術觀察**：隨著迭代次數增加，GPU 的吞吐量優勢（Throughput）呈線性爆發，成功抵銷了 PCIe 頻寬帶來的初始延遲。

## 🛠️ 開發環境
- **Language**: C++ 23
- **Graphics API**: DirectX 12 (Agility SDK)
- **Shader**: HLSL (CS_6_0)
- **IDE**: Visual Studio 2026 (.slnx ready)

## 📂 專案結構
- `API/main.cpp`: 測試邏輯與高精度計時器實作。
- `API/GPU.cpp`: DX12 核心架構、指令列表管理與異步同步邏輯。
- `API/SuperCompute.hlsl`: 最佳化的 GPGPU 計算核心。
- 
##已知問題
* 尚未支持動態調整VRAM(日後版本會做新增)
* 記憶體使用量較大(日後版本會做新增)
* 處理大資料時CPU可能過載(日後版本會做新增)
* 尚未整理成可直接調用的Lib(日後版本會做新增)
* 只支援Windows(暫定沒有要擴展系統)

## 其他
* Demo版本是AI生成的雛形日後會針對此版本錯修改和新增

## 2026 3/26提交
