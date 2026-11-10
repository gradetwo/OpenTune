#include "HNSEPOnnxPostprocessor.h"

#include "../Utils/AccelerationDetector.h"
#include "../Utils/AppLogger.h"
#include "../Utils/SilentGapDetector.h"

#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <dml_provider_factory.h>
#endif

#include "../Inference/OnnxRuntimeProviderCompat.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

namespace OpenTune {
namespace {

constexpr int64_t kHnsepTargetChunkSamples = 5 * 44100;
constexpr int64_t kHnsepMinimumChunkSamples = 3 * 44100;
constexpr int kHnsepHopSize = 512;
constexpr int64_t kHnsepOverlapSamples = 441; // 10ms at the fixed 44.1kHz rate
constexpr int64_t kHnsepLeftOverlapSamples = kHnsepOverlapSamples / 2;

struct InferenceChunk
{
    int64_t startSample{0};
    int64_t endSampleExclusive{0};
};

std::vector<InferenceChunk> planInferenceChunks(
    int64_t sampleCount, const std::vector<SilentGap>& silentGaps)
{
    std::vector<InferenceChunk> chunks;
    std::vector<HnsepSilentGap> ranges;
    ranges.reserve(silentGaps.size());
    for (const auto& gap : silentGaps)
        ranges.push_back({gap.startSample, gap.endSampleExclusive});
    const auto boundaries = buildHnsepChunkBoundaries(
        sampleCount, ranges, kHnsepHopSize,
        kHnsepTargetChunkSamples, kHnsepMinimumChunkSamples);
    if (boundaries.size() < 2)
        return chunks;

    for (size_t i = 0; i + 1 < boundaries.size(); ++i)
        chunks.push_back({boundaries[i], boundaries[i + 1]});

    return chunks;
}

Ort::SessionOptions makeSessionOptions()
{
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    return options;
}

bool tryAppendDml(Ort::SessionOptions& options)
{
#if defined(_WIN32)
    if (AccelerationDetector::getInstance().getSelection().backend
        != AccelerationDetector::AccelBackend::DirectML)
        return false;

    auto& api = Ort::GetApi();
    const OrtDmlApi* dmlApi = nullptr;
    OrtStatus* status = api.GetExecutionProviderApi(
        "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dmlApi));
    if (status != nullptr)
    {
        AppLogger::warn("[HNSEP] DirectML EP unavailable: "
            + juce::String(api.GetErrorMessage(status)));
        api.ReleaseStatus(status);
        return false;
    }
    if (dmlApi == nullptr)
        return false;

    const int adapterIndex = AccelerationDetector::getInstance()
        .getSelection().dmlAdapterIndex;
    status = dmlApi->SessionOptionsAppendExecutionProvider_DML(options, adapterIndex);
    if (status != nullptr)
    {
        AppLogger::warn("[HNSEP] DirectML EP append failed: "
            + juce::String(api.GetErrorMessage(status)));
        api.ReleaseStatus(status);
        return false;
    }
    AppLogger::info("[HNSEP] ONNX session using DirectML adapter "
        + juce::String(adapterIndex));
    return true;
#else
    juce::ignoreUnused(options);
    return false;
#endif
}

bool tryAppendCuda(Ort::SessionOptions& options)
{
#if defined(__linux__)
    if (AccelerationDetector::getInstance().getSelection().backend
        != AccelerationDetector::AccelBackend::CUDA)
        return false;

    const int deviceIndex = AccelerationDetector::getInstance()
        .getSelection().cudaDeviceIndex;
    try
    {
        appendCudaExecutionProvider(options, deviceIndex);
    }
    catch (const std::exception& e)
    {
        AppLogger::warn("[HNSEP] CUDA EP append failed: " + juce::String(e.what()));
        return false;
    }
    AppLogger::info("[HNSEP] ONNX session using CUDA device " + juce::String(deviceIndex));
    return true;
#else
    juce::ignoreUnused(options);
    return false;
#endif
}

std::unique_ptr<Ort::Session> createSession(const std::string& modelPath,
                                             Ort::Env& env,
                                             Ort::SessionOptions& options)
{
#ifdef _WIN32
    const juce::String path(modelPath);
    const auto widePath = path.toWideCharPointer();
    return std::make_unique<Ort::Session>(env, widePath, options);
#else
    return std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
#endif
}

std::vector<float> inferChunk(Ort::Session& session,
                              Ort::MemoryInfo& memoryInfo,
                              const float* dryAudio,
                              int64_t startSample,
                              int64_t endSampleExclusive)
{
    const size_t sampleCount = static_cast<size_t>(endSampleExclusive - startSample);
    std::array<int64_t, 3> shape{1, 1, static_cast<int64_t>(sampleCount)};
    auto input = Ort::Value::CreateTensor<float>(
        memoryInfo,
        const_cast<float*>(dryAudio + startSample),
        sampleCount,
        shape.data(), shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    if (session.GetInputCount() != 1)
        throw std::runtime_error("HNSEP ONNX must expose exactly one waveform input");
    auto inputName = session.GetInputNameAllocated(0, allocator);
    if (inputName == nullptr || std::string(inputName.get()) != "waveform")
        throw std::runtime_error("HNSEP ONNX waveform input must be named 'waveform'");
    const char* inputNames[] = {"waveform"};
    const char* outputNames[] = {"waveform_out"};
    Ort::RunOptions runOptions;
    auto outputs = session.Run(runOptions, inputNames, &input, 1, outputNames, 1);
    if (outputs.size() != 1 || !outputs.front().IsTensor())
        throw std::runtime_error("HNSEP ONNX waveform_out is missing or is not a tensor");

    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    const auto outputShape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || outputShape.size() != 3 || outputShape[0] != 1 || outputShape[1] != 1
        || outputShape[2] != static_cast<int64_t>(sampleCount))
        throw std::runtime_error("HNSEP waveform_out must be float [1,1,N] aligned to its input");

    const float* data = outputs.front().GetTensorMutableData<float>();
    std::vector<float> result(data, data + sampleCount);
    if (!std::all_of(result.begin(), result.end(), [](float value) {
            return std::isfinite(value);
        }))
        throw std::runtime_error("HNSEP waveform_out contains NaN or infinity");
    return result;
}

} // namespace

bool runHNSEPOnnx(const float* dryAudio,
                  size_t sampleCount,
                  const std::string& modelsDir,
                  Ort::Env& env,
                  HNSEPOutput& output,
                  std::string& errorMessage)
{
    output = {};
    errorMessage.clear();
    if (dryAudio == nullptr || sampleCount == 0)
    {
        errorMessage = "HNSEP received no dry audio samples";
        return false;
    }
    if (!std::all_of(dryAudio, dryAudio + sampleCount, [](float value) {
            return std::isfinite(value);
        }))
    {
        errorMessage = "HNSEP dry audio contains NaN or infinity";
        return false;
    }

    static std::atomic<bool> cudaGuidanceLogged{false};
    if (!cudaGuidanceLogged.exchange(true, std::memory_order_acq_rel))
        AppLogger::info("[HNSEP] CUDA-specific HEURISTIC/use_tf32=0 settings: "
            "the CUDA path uses cudnn_conv_algo_search=HEURISTIC when the NVIDIA "
            "driver is present; otherwise DirectML (Windows) or CPU is used, "
            "with sequential single-thread execution");

    const juce::File modelFile = juce::File(juce::String(modelsDir))
        .getChildFile("hnsep.onnx");
    if (!modelFile.existsAsFile())
    {
        errorMessage = "HNSEP ONNX model is missing: "
            + modelFile.getFullPathName().toStdString();
        AppLogger::error("[HNSEP] " + juce::String(errorMessage));
        return false;
    }

    try
    {
        auto options = makeSessionOptions();
        const bool dmlAdded = tryAppendDml(options);
        const bool cudaAdded = dmlAdded ? false : tryAppendCuda(options);
        if (!dmlAdded && !cudaAdded)
            AppLogger::info("[HNSEP] ONNX session using CPU provider");

        std::unique_ptr<Ort::Session> session;
        try
        {
            session = createSession(modelFile.getFullPathName().toStdString(), env, options);
        }
        catch (const std::exception& e)
        {
            if (!dmlAdded && !cudaAdded)
                throw;
            AppLogger::warn("[HNSEP] GPU session creation failed; retrying on CPU: "
                + juce::String(e.what()));
            auto cpuOptions = makeSessionOptions();
            session = createSession(modelFile.getFullPathName().toStdString(), env, cpuOptions);
        }

        juce::AudioBuffer<float> monoAudio(1, static_cast<int>(sampleCount));
        monoAudio.copyFrom(0, 0, dryAudio, static_cast<int>(sampleCount));
        const auto silentGaps = SilentGapDetector::detectAllGapsAdaptive(monoAudio);
        const auto plannedChunks = planInferenceChunks(
            static_cast<int64_t>(sampleCount), silentGaps);
        if (plannedChunks.empty())
            throw std::runtime_error("HNSEP could not plan any non-empty waveform chunks");

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<HnsepChunkOutput> chunkOutputs;
        chunkOutputs.reserve(plannedChunks.size());
        for (size_t i = 0; i < plannedChunks.size(); ++i)
        {
            const auto& core = plannedChunks[i];
            const int64_t inferStart = std::max<int64_t>(0, i == 0
                ? core.startSample
                : core.startSample - kHnsepLeftOverlapSamples);
            const int64_t inferEnd = std::min<int64_t>(static_cast<int64_t>(sampleCount),
                i + 1 == plannedChunks.size()
                    ? core.endSampleExclusive
                    : core.endSampleExclusive
                        + (kHnsepOverlapSamples - kHnsepLeftOverlapSamples));
            chunkOutputs.push_back({inferStart, inferChunk(
                *session, memoryInfo, dryAudio, inferStart, inferEnd)});
        }

        std::vector<float> harmonic;
        if (!stitchHnsepChunkOutputs(sampleCount, chunkOutputs, harmonic))
            throw std::runtime_error("HNSEP chunk stitching did not cover the dry-source timeline");
        std::vector<float> residual;
        if (!makeHnsepResidual(dryAudio, sampleCount, harmonic, residual))
            throw std::runtime_error("HNSEP residual could not be aligned to the dry audio");

        output.harmonic = std::make_shared<const std::vector<float>>(std::move(harmonic));
        output.residual = std::make_shared<const std::vector<float>>(std::move(residual));
        return true;
    }
    catch (const std::exception& e)
    {
        errorMessage = "HNSEP inference failed: " + std::string(e.what());
    }
    catch (...)
    {
        errorMessage = "HNSEP inference failed with an unknown error";
    }

    AppLogger::error("[HNSEP] " + juce::String(errorMessage));
    return false;
}

} // namespace OpenTune
