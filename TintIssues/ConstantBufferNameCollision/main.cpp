#define ENABLE_HLSL

#include <tint/tint.h>
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>
#include <spirv-tools/optimizer.hpp>
#include <spirv-tools/libspirv.hpp>

#include <iostream>
#include <exception>
#include <string>
#include <sstream>

struct TintInitializer
{
    TintInitializer()
    {
        tint::Initialize();
    }
    ~TintInitializer()
    {
        tint::Shutdown();
    }
};

struct GlslangInitilizer
{

    GlslangInitilizer()
    {
        glslang::InitializeProcess();
    }
    ~GlslangInitilizer()
    {
        glslang::FinalizeProcess();
    }
};

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

namespace HLSL
{

const std::string TestVS = R"(
struct Inner
{
    float2x4 Transform;     // 2x4 matrix -> requires a padded layout struct in a uniform buffer
    float4   Offset;
};

cbuffer Params      // source name: "Params"
{
    Inner    g_Data;
    float4x4 g_WorldViewProj;
}

cbuffer Params_1    // source name: literally "Params_1"
{
    float4 g_Color;
}

void main(out float4 Pos : SV_POSITION)
{
    Pos = mul(g_WorldViewProj, g_Data.Offset) + g_Color;
}
)";
;
} // namespace HLSL


std::vector<uint32_t> OptimizeSPIRV(const std::vector<uint32_t>& SrcSPIRV, spv_target_env TargetEnv)
{
    spvtools::Optimizer SpirvOptimizer(TargetEnv);
    SpirvOptimizer.RegisterLegalizationPasses();
    SpirvOptimizer.RegisterPerformancePasses();

    std::vector<uint32_t> OptimizedSPIRV;
    if (!SpirvOptimizer.Run(SrcSPIRV.data(), SrcSPIRV.size(), &OptimizedSPIRV))
        OptimizedSPIRV.clear();

    return OptimizedSPIRV;
}

std::string DisassembleSPIRV(const std::vector<uint32_t>& SPIRV, spv_target_env TargetEnv)
{
    spvtools::SpirvTools SpirvTools(TargetEnv);

    std::string    Disassembly;
    const uint32_t Options = SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_INDENT;
    if (!SpirvTools.Disassemble(SPIRV, &Disassembly, Options))
        LOG_ERROR_AND_THROW("Failed to disassemble SPIR-V.");

    return Disassembly;
}

std::vector<uint32_t> ConvertHLSLtoSPIRV(const std::string& HLSL)
{
    GlslangInitilizer InitScope{};

    glslang::TShader Shader{EShLangVertex};

    auto* pHLSL = HLSL.c_str();
    Shader.setStrings(&pHLSL, 1);
    Shader.setPreamble("#define WEBGPU 1\n");
    Shader.setEntryPoint("main");
    Shader.setEnvInput(glslang::EShSourceHlsl, Shader.getStage(), glslang::EShClientVulkan, 100);
    Shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    Shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
  //  Shader.setEnvTargetHlslFunctionality1();
    Shader.setHlslIoMapping(true);
    Shader.setDxPositionW(true);
    Shader.setAutoMapBindings(true);
    Shader.setAutoMapLocations(true);

    TBuiltInResource Resources{};
    Shader.parse(&Resources, 100, false, EShMsgDefault);

    glslang::TProgram Program;
    Program.addShader(&Shader);

    if (!Program.link(EShMsgDefault))
        LOG_ERROR_AND_THROW("Failed to link program: \n", Program.getInfoLog());

    if (!Program.mapIO())
        LOG_ERROR_AND_THROW("Failed to map IO: \n", Program.getInfoLog());

    std::vector<uint32_t> SPIRV;
    glslang::GlslangToSpv(*Program.getIntermediate(Shader.getStage()), SPIRV);

    auto OptimizedSPIRV = OptimizeSPIRV(SPIRV, SPV_ENV_VULKAN_1_0);

    if (!OptimizedSPIRV.empty())
        return OptimizedSPIRV;

    LOG_ERROR_AND_THROW("Failed to optimize SPIR-V.");
}

std::string ConvertSPIRVtoWGSL(const std::vector<uint32_t>& SPIRV)
{
    TintInitializer InitScope{};

    std::string WGSL;
    auto        Module = tint::spirv::reader::ReadIR(SPIRV);

    if (Module != tint::Success)
        LOG_ERROR_AND_THROW("Tint SPIR-V reader failure:\nParser: ", Module.Failure(), "\n");

    tint::wgsl::writer::Options Options;
    Options.allow_non_uniform_derivatives = true;
    Options.allowed_features              = tint::wgsl::AllowedFeatures::Everything();

    auto Program = tint::wgsl::writer::WgslFromIR(Module.Get(), Options);

    if (Program != tint::Success)
        LOG_ERROR_AND_THROW("Tint WGSL writer failure:\nGenerate: ", Program.Failure().reason, "\n");
    WGSL = std::move(Program.Get().wgsl);
    return WGSL;
}

int main(int argc, const char* argv[])
{
    try
    {
        auto SPIRV = ConvertHLSLtoSPIRV(HLSL::TestVS);
        std::cout << "==== SPIR-V (intermediate) ====\n"
                  << DisassembleSPIRV(SPIRV, SPV_ENV_VULKAN_1_0) << "\n";

        auto WGSL = ConvertSPIRVtoWGSL(SPIRV);
        std::cout << "==== WGSL (Tint output) ====\n"
                  << WGSL << "\n";
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}
