#include "AccelerationDetector.h"
#include "AppLogger.h"
#include <juce_core/juce_core.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <Windows.h>
#include <dxgi.h>
#include <dxgi1_6.h>
#pragma comment(lib, "dxgi.lib")
#include <dml_provider_factory.h>
#endif

#if defined(__linux__)
#include <dlfcn.h>
#include <fstream>
#include <dirent.h>
#include <cstdlib>
#endif

#include <sstream>
#include <iomanip>
#include <onnxruntime_cxx_api.h>
#include <algorithm>

namespace OpenTune {

AccelerationDetector& AccelerationDetector::getInstance() {
    static auto* instance = new AccelerationDetector;
    return *instance;
}

#ifdef _WIN32
bool AccelerationDetector::enumerateGpuDevices(std::vector<GpuDeviceInfo>& gpuDevices) {
    IDXGIFactory1* pFactory = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);
    
    if (FAILED(hr) || pFactory == nullptr) {
        AppLogger::warn("[AccelerationDetector] Failed to create DXGI factory");
        return false;
    }
    
    UINT i = 0;
    IDXGIAdapter1* pAdapter = nullptr;
    
    while (pFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC1 desc;
        pAdapter->GetDesc1(&desc);
        
        GpuDeviceInfo info;
        
        char nameBuf[256];
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, nameBuf, sizeof(nameBuf), nullptr, nullptr);
        info.name = nameBuf;
        
        info.dedicatedVideoMemory = desc.DedicatedVideoMemory;
        info.sharedSystemMemory = desc.SharedSystemMemory;
        info.vendorId = desc.VendorId;
        info.deviceId = desc.DeviceId;
        info.adapterIndex = i;
        
        // 先检查是否为软件渲染器（跳过，不是真实GPU）
        // DXGI_ADAPTER_FLAG_SOFTWARE 标记软件渲染器（如WARP），不是集成显卡
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            pAdapter->Release();
            i++;
            continue;  // 跳过软件渲染器
        }
        
        // 判断是否为集成显卡（基于厂商ID和显存大小）
        info.isIntegrated = false;
        if (desc.VendorId == 0x8086) {  // Intel
            // Intel Arc系列（设备ID 0x5690-0x56C0）是独立显卡，不应标记为集成
            bool isArc = (desc.DeviceId >= 0x5690 && desc.DeviceId <= 0x56C0);
            if (!isArc && desc.DedicatedVideoMemory < 512 * 1024 * 1024) {
                info.isIntegrated = true;
            }
        } else if (desc.VendorId == 0x1002) {  // AMD
            // AMD APU/核显：专用显存通常较小（<1GB）且有共享内存
            // AMD Radeon 760M等核显的VRAM约400-512MB
            // 独立显卡如RX 6600有8GB，RX 580有4GB
            if (desc.DedicatedVideoMemory < 1024 * 1024 * 1024 && desc.SharedSystemMemory > 0) {
                info.isIntegrated = true;
            }
        }
        
        // 检查是否为真正的 GPU（排除软件渲染器和无效设备）
        if (info.vendorId != 0 && !info.name.empty() && info.name != "Microsoft Basic Render Driver") {
            gpuDevices.push_back(info);
            
            // 格式化显存大小
            float vramGB = static_cast<float>(info.dedicatedVideoMemory) / (1024.0f * 1024.0f * 1024.0f);
            AppLogger::debug("[AccelerationDetector] Found GPU: " + juce::String(info.name) 
                      + " (" + juce::String(vramGB, 1) + " GB VRAM)");
        }
        
        pAdapter->Release();
        i++;
    }
    
    pFactory->Release();
    
    return !gpuDevices.empty();
}
#else
bool AccelerationDetector::enumerateGpuDevices(std::vector<GpuDeviceInfo>& gpuDevices) {
#if defined(__linux__)
    // Linux：从 NVIDIA 驱动导出的 /proc 接口枚举 GPU。
    // 每个物理 GPU 对应 /proc/driver/nvidia/gpus/<pci>/information，
    // 内含 Model / Driver Version 字段。无需 NVML 依赖。
    DIR* dir = opendir("/proc/driver/nvidia/gpus");
    if (dir == nullptr) {
        AppLogger::debug("[AccelerationDetector] /proc/driver/nvidia/gpus not available (no NVIDIA driver)");
        return false;
    }

    struct dirent* entry = nullptr;
    uint32_t deviceIndex = 0;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.')
            continue;

        const std::string infoPath = std::string("/proc/driver/nvidia/gpus/")
            + entry->d_name + "/information";
        std::ifstream infoFile(infoPath);
        if (!infoFile.is_open())
            continue;

        GpuDeviceInfo info;
        std::string line;
        while (std::getline(infoFile, line)) {
            const auto readField = [&line](const char* key) -> std::string {
                const std::string prefix = key;
                if (line.rfind(prefix, 0) != 0)
                    return {};
                std::string value = line.substr(prefix.size());
                while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
                    value.erase(value.begin());
                return value;
            };
            if (auto model = readField("Model : "); !model.empty())
                info.name = model;
            else if (auto irq = readField("IRQ : "); !irq.empty())
                info.deviceId = static_cast<size_t>(std::strtoull(irq.c_str(), nullptr, 0));
            else if (auto guid = readField("GPU UUID : "); !guid.empty())
                info.vendorId = 0x10DE; // NVIDIA PCI vendor id
        }

        if (info.name.empty())
            continue;

        info.vendorId = 0x10DE;
        info.adapterIndex = deviceIndex++;
        info.isIntegrated = false;
        gpuDevices.push_back(info);
        AppLogger::debug("[AccelerationDetector] Found GPU: " + juce::String(info.name)
                  + " (NVIDIA, device " + juce::String(static_cast<int>(info.adapterIndex)) + ")");
    }
    closedir(dir);

    return !gpuDevices.empty();
#else
    juce::ignoreUnused(gpuDevices);
    return false;
#endif
}
#endif

bool AccelerationDetector::detectDirectML(GpuDeviceInfo& selectedGpu, int& adapterIndex) {
#ifdef _WIN32
    std::vector<GpuDeviceInfo> gpuDevices;
    if (!enumerateGpuDevices(gpuDevices)) {
        AppLogger::warn("[AccelerationDetector] No DirectX 12 compatible GPU found");
        return false;
    }

    // 选择最佳 GPU（独显优先，VRAM 降序）— 仅用于日志/UI 和 adapterIndex
    std::sort(gpuDevices.begin(), gpuDevices.end(), [](const GpuDeviceInfo& a, const GpuDeviceInfo& b) {
        if (a.isIntegrated != b.isIntegrated) return !a.isIntegrated;
        return a.dedicatedVideoMemory > b.dedicatedVideoMemory;
    });
    selectedGpu = gpuDevices[0];
    adapterIndex = static_cast<int>(selectedGpu.adapterIndex);

    // 用 ORT API 判断 DML EP 是否编译进当前 OpenTuneOnnxRuntime_1_24_4.dll
    Ort::InitApi();
    auto& api = Ort::GetApi();
    const OrtDmlApi* dmlApi = nullptr;
    OrtStatus* status = api.GetExecutionProviderApi(
        "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dmlApi));

    if (status != nullptr) {
        std::string msg = api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        AppLogger::warn("[AccelerationDetector] DML EP not available in this ORT build: " + juce::String(msg));
        return false;
    }

    if (dmlApi == nullptr) {
        AppLogger::warn("[AccelerationDetector] DML EP API pointer is null");
        return false;
    }

    float vramGB = static_cast<float>(selectedGpu.dedicatedVideoMemory) / (1024.0f * 1024.0f * 1024.0f);
    AppLogger::info("[AccelerationDetector] DML EP available, selected GPU: " + juce::String(selectedGpu.name)
                      + " (" + juce::String(vramGB, 1) + " GB VRAM, adapterIndex="
                      + juce::String(static_cast<int>(selectedGpu.adapterIndex)) + ")");
    return true;
#else
    return false;
#endif
}

bool AccelerationDetector::detectCoreML() {
#if defined(__APPLE__)
    // CoreML 是 macOS 系统框架，在所有 macOS 版本上都可用
    // 实际 EP 兼容性（算子支持等）由 ModelFactory::createSessionOptions 的 try/catch 处理
    AppLogger::info("[AccelerationDetector] CoreML available (macOS system framework)");
    return true;
#else
    AppLogger::debug("[AccelerationDetector] CoreML only available on macOS");
    return false;
#endif
}

bool AccelerationDetector::detectCuda() {
#if defined(__linux__)
    // 1) NVIDIA 驱动必须存在：枚举 /proc/driver/nvidia/gpus，
    //    失败时回退到 dlopen("libcuda.so.1") 探测。
    std::vector<GpuDeviceInfo> gpuDevices;
    if (!enumerateGpuDevices(gpuDevices)) {
        void* driverHandle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (driverHandle == nullptr)
            driverHandle = dlopen("libcuda.so.1", RTLD_NOW);
        if (driverHandle == nullptr) {
            AppLogger::debug("[AccelerationDetector] No NVIDIA driver (libcuda.so.1 not loadable), CUDA unavailable");
            return false;
        }
        dlclose(driverHandle);
        AppLogger::warn("[AccelerationDetector] libcuda.so.1 found but no /proc GPU info; assuming NVIDIA GPU present");
    }

    // 2) 随包 ONNX Runtime 必须编译了 CUDA EP（官方 linux-x64 包内置）。
    //    GetExecutionProviderApi("CUDA") 会验证 provider 动态库可加载且 ABI 匹配。
    Ort::InitApi();
    auto& api = Ort::GetApi();
    const void* cudaProviderApi = nullptr;
    OrtStatus* status = api.GetExecutionProviderApi(
        "CUDA", ORT_API_VERSION, &cudaProviderApi);
    if (status != nullptr) {
        const std::string message = api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        AppLogger::warn("[AccelerationDetector] CUDA EP not available in this ONNX Runtime build: "
                          + juce::String(message));
        return false;
    }
    if (cudaProviderApi == nullptr) {
        AppLogger::warn("[AccelerationDetector] CUDA EP API pointer is null");
        return false;
    }

    GpuDeviceInfo selected;
    if (!gpuDevices.empty()) {
        selected = gpuDevices.front();
        AppLogger::info("[AccelerationDetector] CUDA EP available, selected GPU: " + juce::String(selected.name)
                          + " (device " + juce::String(static_cast<int>(selected.adapterIndex)) + ")");
    } else {
        AppLogger::info("[AccelerationDetector] CUDA EP available, NVIDIA driver detected");
    }
    return true;
#else
    AppLogger::debug("[AccelerationDetector] CUDA only available on Linux");
    return false;
#endif
}

void AccelerationDetector::detect(bool forceCpu) {
    if (!forceCpu && selection_.load(std::memory_order_acquire) != kUndetected)
        return;

    const auto selection = detectSelection(forceCpu);
    const auto encoded = encodeSelection(selection);

    if (forceCpu) {
        selection_.store(encoded, std::memory_order_release);
        return;
    }

    uint64_t expected = kUndetected;
    selection_.compare_exchange_strong(expected, encoded,
                                       std::memory_order_release,
                                       std::memory_order_acquire);
}

void AccelerationDetector::resetAndDetect(bool forceCpu) {
    const auto selection = detectSelection(forceCpu);
    selection_.store(encodeSelection(selection), std::memory_order_release);
}

AccelerationDetector::BackendSelection AccelerationDetector::detectSelection(bool forceCpu) {
    BackendSelection selection;

    AppLogger::debug("[AccelerationDetector] Detecting acceleration capabilities...");

    if (forceCpu) {
        AppLogger::info("[AccelerationDetector] CPU forced by user preference");
        AppLogger::info("[AccelerationDetector] Selected backend: CPU");
        return selection;
    }

#if defined(__APPLE__)
    if (detectCoreML()) {
        selection.backend = AccelBackend::CoreML;
    } else {
        AppLogger::info("[AccelerationDetector] No acceleration available, using CPU");
    }
#elif defined(_WIN32)
    GpuDeviceInfo selectedGpu;
    if (detectDirectML(selectedGpu, selection.dmlAdapterIndex)) {
        selection.backend = AccelBackend::DirectML;
    } else {
        AppLogger::info("[AccelerationDetector] No GPU acceleration available, using CPU");
    }
#elif defined(__linux__)
    if (detectCuda()) {
        std::vector<GpuDeviceInfo> gpuDevices;
        enumerateGpuDevices(gpuDevices);
        if (!gpuDevices.empty())
            selection.cudaDeviceIndex = static_cast<int>(gpuDevices.front().adapterIndex);
        selection.backend = AccelBackend::CUDA;
    } else {
        AppLogger::info("[AccelerationDetector] No CUDA acceleration available, using CPU");
    }
#else
    AppLogger::info("[AccelerationDetector] No acceleration available, using CPU");
#endif

    AppLogger::info("[AccelerationDetector] Selected backend: "
        + juce::String(backendName(selection.backend)));
    return selection;
}

void AccelerationDetector::overrideBackend(AccelBackend backend) {
    auto selection = getSelection();
    selection.backend = backend;
    if (backend != AccelBackend::DirectML)
        selection.dmlAdapterIndex = 0;
    if (backend != AccelBackend::CUDA)
        selection.cudaDeviceIndex = 0;
    selection_.store(encodeSelection(selection), std::memory_order_release);
}

AccelerationDetector::BackendSelection AccelerationDetector::getSelection() const noexcept {
    return decodeSelection(selection_.load(std::memory_order_acquire));
}

const char* AccelerationDetector::backendName(AccelBackend backend) noexcept {
    switch (backend) {
        case AccelBackend::CoreML: return "CoreML";
        case AccelBackend::DirectML: return "DirectML";
        case AccelBackend::CUDA: return "CUDA";
        case AccelBackend::CPU: return "CPU";
    }
    return "Unknown";
}

uint64_t AccelerationDetector::encodeSelection(BackendSelection selection) noexcept {
    return (static_cast<uint64_t>(selection.backend) << 32)
        | (static_cast<uint64_t>(static_cast<uint32_t>(selection.cudaDeviceIndex) & 0xffffu) << 16)
        | static_cast<uint32_t>(selection.dmlAdapterIndex);
}

AccelerationDetector::BackendSelection AccelerationDetector::decodeSelection(uint64_t encoded) noexcept {
    if (encoded == kUndetected)
        return {};

    BackendSelection selection;
    selection.backend = static_cast<AccelBackend>((encoded >> 32) & 0xffu);
    selection.cudaDeviceIndex = static_cast<int>((encoded >> 16) & 0xffffu);
    selection.dmlAdapterIndex = static_cast<int>(static_cast<uint32_t>(encoded));
    return selection;
}

} // namespace OpenTune
