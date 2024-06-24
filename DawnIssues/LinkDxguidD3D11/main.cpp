#define ENABLE_HLSL

#include <iostream>
#include <exception>
#include <vector>
#include <string>
#include <sstream>

#include <webgpu/webgpu.h>
#include "WebGPUObjectWrapper.hpp"

template <typename... Args>
std::string ConcatenateArgs(Args... args)
{
    std::ostringstream OutputStream;
    (OutputStream << ... << args); // Fold expression to concatenate all arguments
    return OutputStream.str();
}

#define LOG_ERROR_AND_THROW(...)                                        \
    do                                                                  \
    {                                                                   \
        std::string Message = ConcatenateArgs(__VA_ARGS__);             \
        std::cerr << "Error: " << Message << " (in " << __FILE__ << ":" \
                  << __LINE__ << ", function " << __func__ << ")"       \
                  << std::endl;                                         \
        throw std::runtime_error(Message);                              \
    } while (false)

#define LOG_WARNING_MESSAGE(...)                            \
    do                                                      \
    {                                                       \
        std::string Message = ConcatenateArgs(__VA_ARGS__); \
        std::cerr << "Warning: " << Message << std::endl;   \
    } while (false)

#define LOG_INFO_MESSAGE(...)                               \
    do                                                      \
    {                                                       \
        std::string Message = ConcatenateArgs(__VA_ARGS__); \
        std::cerr << "Info: " << Message << std::endl;      \
    } while (false)

std::vector<WebGPUAdapterWrapper> FindCompatibleAdapters(WGPUInstance wgpuInstance)
{
    std::vector<WebGPUAdapterWrapper> wgpuAdapters;

    struct CallbackUserData
    {
        WGPUAdapter              Adapter       = nullptr;
        WGPURequestAdapterStatus RequestStatus = {};
        std::string              Message       = {};
    };

    auto OnAdapterRequestEnded = [](WGPURequestAdapterStatus Status, WGPUAdapter Adapter, char const* Message, void* pCallbackUserData) {
        auto* pUserData          = static_cast<CallbackUserData*>(pCallbackUserData);
        pUserData->Adapter       = Adapter;
        pUserData->RequestStatus = Status;
        if (Message != nullptr)
            pUserData->Message = Message;
    };

    WGPUPowerPreference PowerPreferences[] = {
        WGPUPowerPreference_HighPerformance,
        WGPUPowerPreference_LowPower};

    for (const auto& powerPreference : PowerPreferences)
    {
        CallbackUserData UserData{};

        WGPURequestAdapterOptions Options{nullptr, nullptr, powerPreference, WGPUBackendType_Undefined, false};
        wgpuInstanceRequestAdapter(
            wgpuInstance,
            &Options,
            OnAdapterRequestEnded,
            &UserData);

        if (UserData.RequestStatus == WGPURequestAdapterStatus_Success)
        {
            auto IsFound = std::find_if(wgpuAdapters.begin(), wgpuAdapters.end(),
                                        [&](const auto& wgpuAdapter) { return wgpuAdapter.Get() == UserData.Adapter; });

            if (IsFound == wgpuAdapters.end())
                wgpuAdapters.emplace_back(UserData.Adapter);
        }
        else
        {
            LOG_WARNING_MESSAGE(UserData.Message);
        }
    }
    return wgpuAdapters;
}

int main(int argc, const char* argv[])
{
    try
    {
        WGPUInstanceDescriptor wgpuInstanceDesc = {};
        WebGPUInstanceWrapper  wgpuInstance{wgpuCreateInstance(&wgpuInstanceDesc)};
        if (!wgpuInstance)
            LOG_ERROR_AND_THROW("Failed to create WebGPU instance");

        auto wgpuAdapters = FindCompatibleAdapters(wgpuInstance.Get());

        for (auto const& wgpuAdapter : wgpuAdapters)
        {
            WGPUAdapterProperties wgpuAdapterProperties{};
            wgpuAdapterGetProperties(wgpuAdapter.Get(), &wgpuAdapterProperties);
            LOG_INFO_MESSAGE("Adapter name: ", wgpuAdapterProperties.name);
        }
    }
    catch (const std::exception&)
    {
        return -1;
    }
}
