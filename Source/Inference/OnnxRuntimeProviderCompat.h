#pragma once

#if defined(__APPLE__)

#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>

#include <stdexcept>
#include <string>

namespace OpenTune {

/**
    macOS 12 uses the older Core ML NeuralNetwork model format. It is supported
    by the macOS 12 Core ML runtime and avoids making the first-generation
    MLProgram path a deployment requirement for the oldest supported system.
    Newer systems use MLProgram and all available Core ML compute units so that
    Apple Neural Engine devices (including M-series Macs) are not limited to
    CPU/GPU.
*/
inline bool runningOnMacOs12()
{
#if defined(__APPLE__)
    // The application deployment target is macOS 12.0, so "not macOS 13"
    // precisely identifies the macOS 12 compatibility path.
    if (__builtin_available(macOS 13.0, *))
        return false;
    return true;
#else
    return false;
#endif
}

inline const char* coreMlProviderDescription()
{
    return runningOnMacOs12() ? "NeuralNetwork/macOS 12" : "MLProgram/macOS 13+";
}

/**
    Registers the CoreML execution provider on the given session options.

    The public CoreML API in the ORT 1.20.x package is the flag-based C API.
    The default flag set selects the NeuralNetwork format and all available
    compute units; MLProgram is explicitly enabled only on macOS 13+.
*/
inline void appendCoreMlExecutionProvider(Ort::SessionOptions& sessionOptions)
{
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const uint32_t coremlFlags = runningOnMacOs12()
        ? COREML_FLAG_USE_NONE
        : COREML_FLAG_CREATE_MLPROGRAM;
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(
        sessionOptions, coremlFlags);
    if (status != nullptr)
    {
        const std::string message = api->GetErrorMessage(status);
        api->ReleaseStatus(status);
        throw std::runtime_error(message);
    }
}

} // namespace OpenTune

#elif defined(__linux__)

#include <onnxruntime_cxx_api.h>

#include <stdexcept>
#include <string>

namespace OpenTune {

/**
    Registers the CUDA execution provider on the given session options (Linux).

    The CUDA EP lives in libonnxruntime_providers_cuda.so shipped next to
    libonnxruntime. Registering it only succeeds when an NVIDIA driver and a
    CUDA/cuDNN runtime are installed on the host; otherwise the returned
    OrtStatus carries the reason and callers fall back to CPU.

    Prefers the CUDA provider options V2 API (device selection, cuDNN algo
    search policy) and falls back to the legacy V1 entry point when the V2
    pointers are absent from this header/version of ONNX Runtime.
*/
inline void appendCudaExecutionProvider(Ort::SessionOptions& sessionOptions,
                                        int deviceId = 0)
{
    Ort::InitApi();
    const OrtApi& api = Ort::GetApi();

    // Preferred path: opaque OrtCUDAProviderOptionsV2 + string key/value update.
    if (api.CreateCUDAProviderOptions != nullptr
            && api.UpdateCUDAProviderOptions != nullptr
            && api.SessionOptionsAppendExecutionProvider_CUDA_V2 != nullptr
            && api.ReleaseCUDAProviderOptions != nullptr)
    {
        OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
        OrtStatus* status = api.CreateCUDAProviderOptions(&cudaOptions);
        if (status != nullptr)
        {
            const std::string message = api.GetErrorMessage(status);
            api.ReleaseStatus(status);
            throw std::runtime_error("CreateCUDAProviderOptions failed: " + message);
        }

        const std::string deviceIdValue = std::to_string(deviceId);
        const char* keys[] = { "device_id", "cudnn_conv_algo_search" };
        const char* values[] = { deviceIdValue.c_str(), "HEURISTIC" };
        status = api.UpdateCUDAProviderOptions(cudaOptions, keys, values, 2);
        if (status != nullptr)
        {
            const std::string message = api.GetErrorMessage(status);
            api.ReleaseStatus(status);
            api.ReleaseCUDAProviderOptions(cudaOptions);
            throw std::runtime_error("UpdateCUDAProviderOptions failed: " + message);
        }

        status = api.SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cudaOptions);
        api.ReleaseCUDAProviderOptions(cudaOptions);
        if (status != nullptr)
        {
            const std::string message = api.GetErrorMessage(status);
            api.ReleaseStatus(status);
            throw std::runtime_error(message);
        }
        return;
    }

    // Legacy V1 fallback: public OrtCUDAProviderOptions struct + V1 append call.
    if (api.SessionOptionsAppendExecutionProvider_CUDA == nullptr)
        throw std::runtime_error("CUDA EP APIs are not available in this ONNX Runtime build");

    OrtCUDAProviderOptions legacyOptions {};
    legacyOptions.device_id = deviceId;
    OrtStatus* status = api.SessionOptionsAppendExecutionProvider_CUDA(
        sessionOptions, &legacyOptions);
    if (status != nullptr)
    {
        const std::string message = api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        throw std::runtime_error(message);
    }
}

inline const char* cudaProviderDescription()
{
    return "CUDA EP (NVIDIA GPU)";
}

} // namespace OpenTune

#endif // __APPLE__ / __linux__
