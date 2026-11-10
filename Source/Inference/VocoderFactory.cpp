#include "VocoderFactory.h"
#include "PCNSFHifiGANVocoder.h"
#include "OnnxRuntimeProviderCompat.h"
#include "VocoderSidecarMetadata.h"
#ifdef _WIN32
#include "DmlVocoder.h"
#endif
#include "../Utils/AccelerationDetector.h"
#include "../Utils/AppLogger.h"
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace OpenTune {

namespace {

struct SidecarConfig
{
    std::optional<float> melFMax;
    std::optional<float> melClipVal;
    std::optional<MelFilterbankSpec> melFilterbank;
    bool needsHighFrequencyOriginal{false};
};

std::string trimAscii(std::string value)
{
    const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && isSpace(static_cast<unsigned char>(value.front())))
        value.erase(value.begin());
    while (!value.empty() && isSpace(static_cast<unsigned char>(value.back())))
        value.pop_back();
    return value;
}

float parseYamlFloat(const std::string& value, const char* field)
{
    try {
        size_t consumed = 0;
        const float result = std::stof(trimAscii(value), &consumed);
        if (consumed != trimAscii(value).size() || !(result > 0.0f))
            throw std::runtime_error("not a positive number");
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid sidecar field ") + field);
    }
}

int parseYamlInt(const std::string& value, const char* field)
{
    try {
        size_t consumed = 0;
        const auto text = trimAscii(value);
        const int result = std::stoi(text, &consumed);
        if (consumed != text.size() || result <= 0)
            throw std::runtime_error("not a positive integer");
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid sidecar field ") + field);
    }
}

SidecarConfig loadSidecarConfig(const std::string& modelPath)
{
    const juce::File yamlFile = juce::File(modelPath).withFileExtension("yaml");
    if (!yamlFile.existsAsFile())
        return {};

    const auto lines = juce::StringArray::fromLines(yamlFile.loadFileAsString());
    SidecarConfig result;
    result.needsHighFrequencyOriginal = sidecarNeedsHighFrequencyOriginal(lines);
    MelFilterbankSpec filterbank;
    std::vector<float> currentRegion;
    bool inFilterbank = false;
    bool inRegions = false;
    bool filterbankSeen = false;
    bool filterbankTypeSeen = false;

    const auto finishRegion = [&]() {
        if (currentRegion.empty())
            return;
        if (currentRegion.size() != 3)
            throw std::runtime_error("Invalid high_frequency_bin_counts region in sidecar");
        filterbank.regions.push_back({
            currentRegion[0], currentRegion[1], static_cast<int>(currentRegion[2])
        });
        currentRegion.clear();
    };

    for (const auto& line : lines)
    {
        const auto raw = line.toStdString();
        const auto trimmed = trimAscii(raw);
        if (trimmed.empty() || trimmed.front() == '#')
            continue;

        const bool topLevel = !raw.empty()
            && !std::isspace(static_cast<unsigned char>(raw.front()));
        if (inFilterbank && topLevel)
        {
            finishRegion();
            inFilterbank = false;
            inRegions = false;
        }

        if (!inFilterbank)
        {
            if (topLevel && trimmed.rfind("mel_fmax:", 0) == 0)
                result.melFMax = parseYamlFloat(trimmed.substr(9), "mel_fmax");
            else if (topLevel && trimmed.rfind("mel_clip_val:", 0) == 0)
                result.melClipVal = parseYamlFloat(trimmed.substr(13), "mel_clip_val");
            else if (topLevel && trimmed == "mel_filterbank:")
            {
                inFilterbank = true;
                inRegions = false;
                filterbankSeen = true;
            }
            continue;
        }

        if (trimmed.rfind("type:", 0) == 0)
        {
            const auto type = trimAscii(trimmed.substr(5));
            if (type != "custom_high_frequency")
                throw std::runtime_error("Unsupported mel_filterbank.type in sidecar: " + type);
            filterbank.type = MelFilterbankSpec::Type::CustomHighFrequency;
            filterbankTypeSeen = true;
        }
        else if (trimmed.rfind("base_num_bins:", 0) == 0)
        {
            filterbank.baseNumBins = parseYamlInt(trimmed.substr(14), "base_num_bins");
        }
        else if (trimmed == "high_frequency_bin_counts:")
        {
            inRegions = true;
        }
        else if (inRegions && trimmed.rfind("- -", 0) == 0)
        {
            finishRegion();
            currentRegion.push_back(parseYamlFloat(trimmed.substr(3), "high_frequency_bin_counts.low"));
        }
        else if (inRegions && trimmed.front() == '-')
        {
            if (currentRegion.size() >= 3)
                throw std::runtime_error("Too many values in high_frequency_bin_counts region");
            if (currentRegion.size() == 2)
                currentRegion.push_back(static_cast<float>(parseYamlInt(
                    trimmed.substr(1), "high_frequency_bin_counts.count")));
            else
                currentRegion.push_back(parseYamlFloat(
                    trimmed.substr(1), "high_frequency_bin_counts"));
        }
    }

    if (inFilterbank)
        finishRegion();
    if (filterbankSeen)
    {
        if (!filterbankTypeSeen || !filterbank.isCustom()
            || filterbank.baseNumBins <= 0 || filterbank.regions.empty())
        {
            throw std::runtime_error("Incomplete custom mel_filterbank sidecar configuration");
        }
        result.melFilterbank = std::move(filterbank);
    }
    return result;
}

void applySidecarConfig(const std::string& modelPath, VocoderInterface& vocoder)
{
    const auto config = loadSidecarConfig(modelPath);
    vocoder.setNeedsHighFrequencyOriginal(config.needsHighFrequencyOriginal);
    if (config.melFMax.has_value()) {
        vocoder.setMelFMax(*config.melFMax);
        AppLogger::info("[VocoderFactory] Sidecar mel_fmax=" + juce::String(*config.melFMax)
            + " (" + juce::File(modelPath).getFileName() + ".yaml)");
    }
    if (config.melClipVal.has_value())
        vocoder.setMelLogEps(*config.melClipVal);
    if (config.melFilterbank.has_value())
    {
        vocoder.setMelFilterbankSpec(*config.melFilterbank);
        AppLogger::info("[VocoderFactory] Sidecar custom high-frequency mel filterbank enabled"
            " (base=" + juce::String(config.melFilterbank->baseNumBins)
            + ", regions=" + juce::String(static_cast<int>(config.melFilterbank->regions.size())) + ")");
    }
}

} // namespace

VocoderCreationResult VocoderFactory::create(
    const std::string& modelPath,
    Ort::Env& env)
{
    std::string backendFallbackError;
#ifdef _WIN32
    const auto backendSelection = AccelerationDetector::getInstance().getSelection();

    if (backendSelection.backend == AccelerationDetector::AccelBackend::DirectML) {
        const int adapterIndex = backendSelection.dmlAdapterIndex;

        AppLogger::info("[VocoderFactory] DML backend selected by GPU detector");
        AppLogger::info("[VocoderFactory]   adapterIndex=" + juce::String(adapterIndex));

        AppLogger::info("[VocoderFactory] Creating DML vocoder...");

        try {
            auto vocoder = std::make_unique<DmlVocoder>(modelPath, env, adapterIndex);

            AppLogger::info("[VocoderFactory] DML vocoder created successfully");

            applySidecarConfig(modelPath, *vocoder);

            const float fmax = vocoder->getFMax();
            const float nyquist = static_cast<float>(vocoder->getSampleRate()) * 0.5f;
            if (fmax > nyquist) {
                return VocoderCreationResult::failure(VocoderBackend::DML,
                    "Vocoder config violates Nyquist: fmax=" + std::to_string(fmax)
                    + " > Nyquist=" + std::to_string(nyquist)
                    + " (sampleRate=" + std::to_string(vocoder->getSampleRate()) + ")");
            }
            AppLogger::info("[VocoderFactory] Vocoder config OK: fmax=" + juce::String(fmax)
                + ", sampleRate=" + juce::String(vocoder->getSampleRate())
                + " (Nyquist=" + juce::String(nyquist) + ")");

            return VocoderCreationResult::success(std::move(vocoder), VocoderBackend::DML);

        } catch (const std::exception& e) {
            backendFallbackError = "DirectML initialization failed: " + std::string(e.what());
            AppLogger::error("[VocoderFactory] DML vocoder creation FAILED: "
                + juce::String(e.what()));
            AppLogger::warn("[VocoderFactory] GPU DML initialization failed, switching to CPU vocoder");
            AccelerationDetector::getInstance().overrideBackend(AccelerationDetector::AccelBackend::CPU);
        }
    }
#endif
    
    try {
        Ort::SessionOptions sessionOptions;

        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        VocoderBackend selectedBackend = VocoderBackend::CPU;

#if defined(__APPLE__)
        // macOS 12 deliberately uses the NeuralNetwork Core ML format. Newer
        // systems use MLProgram with all available compute units.
        if (AccelerationDetector::getInstance().getSelection().backend
                == AccelerationDetector::AccelBackend::CoreML) {
            try {
                appendCoreMlExecutionProvider(sessionOptions);
                selectedBackend = VocoderBackend::CoreML;
                AppLogger::info("[VocoderFactory] Vocoder session: CoreML EP added ("
                    + juce::String(coreMlProviderDescription()) + ", computeUnits=ALL)");
            } catch (const std::exception& e) {
                backendFallbackError = "CoreML execution provider initialization failed: "
                    + std::string(e.what());
                AppLogger::warn(juce::String("[VocoderFactory] Failed to add CoreML EP for vocoder: ") + e.what());
                AppLogger::info("[VocoderFactory] Vocoder session: falling back to CPU");
                AccelerationDetector::getInstance().overrideBackend(AccelerationDetector::AccelBackend::CPU);
            }
        }
#endif

#if defined(__linux__)
        if (AccelerationDetector::getInstance().getSelection().backend
                == AccelerationDetector::AccelBackend::CUDA) {
            const int cudaDeviceIndex = AccelerationDetector::getInstance()
                .getSelection().cudaDeviceIndex;
            try {
                appendCudaExecutionProvider(sessionOptions, cudaDeviceIndex);
                selectedBackend = VocoderBackend::CUDA;
                AppLogger::info("[VocoderFactory] Vocoder session: CUDA EP added (device "
                    + juce::String(cudaDeviceIndex) + ", cudnn_conv_algo_search=HEURISTIC)");
            } catch (const std::exception& e) {
                backendFallbackError = "CUDA execution provider initialization failed: "
                    + std::string(e.what());
                AppLogger::warn(juce::String("[VocoderFactory] Failed to add CUDA EP for vocoder: ") + e.what());
                AppLogger::info("[VocoderFactory] Vocoder session: falling back to CPU");
                AccelerationDetector::getInstance().overrideBackend(AccelerationDetector::AccelBackend::CPU);
            }
        }
#endif

        if (selectedBackend == VocoderBackend::CPU) {
            AppLogger::info("[VocoderFactory] Creating CPU vocoder...");
        } else {
            AppLogger::info("[VocoderFactory] Creating "
                + juce::String(selectedBackend == VocoderBackend::CoreML ? "CoreML" : "CUDA")
                + " vocoder...");
        }

        auto createSession = [&env, &modelPath](Ort::SessionOptions& options) {
#ifdef _WIN32
            juce::String jucePath(modelPath);
            auto wPath = jucePath.toWideCharPointer();
            return std::make_unique<Ort::Session>(env, wPath, options);
#else
            return std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
#endif
        };

        std::unique_ptr<Ort::Session> session;
        try {
            session = createSession(sessionOptions);
        } catch (const std::exception& e) {
            if (selectedBackend == VocoderBackend::CPU)
                throw;

            backendFallbackError = (selectedBackend == VocoderBackend::CoreML
                    ? "CoreML session creation failed: "
                    : "CUDA session creation failed: ") + std::string(e.what());
            AppLogger::warn(juce::String("[VocoderFactory] GPU session creation failed: ") + e.what());
            AppLogger::info("[VocoderFactory] Retrying vocoder with CPU");
            AccelerationDetector::getInstance().overrideBackend(AccelerationDetector::AccelBackend::CPU);

            Ort::SessionOptions cpuOptions;
            cpuOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            cpuOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session = createSession(cpuOptions);
            selectedBackend = VocoderBackend::CPU;
        }
        
        auto vocoder = std::make_unique<PCNSFHifiGANVocoder>(std::move(session));

        const juce::String backendStr =
            selectedBackend == VocoderBackend::CoreML ? "CoreML"
            : selectedBackend == VocoderBackend::CUDA ? "CUDA" : "CPU";
        AppLogger::info("[VocoderFactory] " + backendStr + " vocoder created successfully");

        applySidecarConfig(modelPath, *vocoder);

        const float fmax = vocoder->getFMax();
        const float nyquist = static_cast<float>(vocoder->getSampleRate()) * 0.5f;
        if (fmax > nyquist) {
            return VocoderCreationResult::failure(selectedBackend,
                "Vocoder config violates Nyquist: fmax=" + std::to_string(fmax)
                + " > Nyquist=" + std::to_string(nyquist)
                + " (sampleRate=" + std::to_string(vocoder->getSampleRate()) + ")");
        }
        AppLogger::info("[VocoderFactory] Vocoder config OK: fmax=" + juce::String(fmax)
            + ", sampleRate=" + juce::String(vocoder->getSampleRate())
            + " (Nyquist=" + juce::String(nyquist) + ")");

        return VocoderCreationResult::success(std::move(vocoder), selectedBackend);
        
    } catch (const std::exception& e) {
        const auto prefix = backendFallbackError.empty()
            ? std::string{}
            : backendFallbackError + "; ";
        return VocoderCreationResult::failure(VocoderBackend::CPU,
            prefix + "Vocoder creation failed: " + std::string(e.what()));
    }
}

} // namespace OpenTune
