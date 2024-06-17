#define ENABLE_HLSL

#include <tint/tint.h>
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>
#include <spirv-tools/optimizer.hpp>

#include <iostream>
#include <exception>

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

const std::string FillTextureCS = R"(

RWTexture2D</*format=rgba8*/ float4> g_tex2DUAV : register(u0);
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	uint2 ui2Dim;
	g_tex2DUAV.GetDimensions(ui2Dim.x, ui2Dim.y);
	if (DTid.x >= ui2Dim.x || DTid.y >= ui2Dim.y)
        return;

	g_tex2DUAV[DTid.xy] = float4(float2(DTid.xy % 256u) / 256.0, 0.0, 1.0);
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

std::vector<uint32_t> ConvertHLSLtoSPIRV(const std::string& HLSL)
{
    GlslangInitilizer InitScope{};

    glslang::TShader Shader{EShLangCompute};

    auto* pHLSL = HLSL.c_str();
    Shader.setStrings(&pHLSL, 1);
    Shader.setEntryPoint("main");
    Shader.setEnvInput(glslang::EShSourceHlsl, Shader.getStage(), glslang::EShClientVulkan, 100);
    Shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    Shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    Shader.setEnvTargetHlslFunctionality1();
    Shader.setHlslIoMapping(true);
    Shader.setDxPositionW(true);

    TBuiltInResource Resources{};
    Shader.parse(&Resources, 100, false, EShMsgDefault);

    glslang::TProgram Program;
    Program.addShader(&Shader);

    if (!Program.link(EShMsgDefault))
        LOG_ERROR_AND_THROW("Failed to link program: \n", Program.getInfoLog());

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

    tint::wgsl::writer::ProgramOptions Options;
    Options.allow_non_uniform_derivatives = true;
    Options.allowed_features              = tint::wgsl::AllowedFeatures::Everything();

    auto Program = tint::wgsl::writer::WgslFromIR(Module.Get(), Options);

    if (Program != tint::Success)
        LOG_ERROR_AND_THROW("Tint WGSL writer failure:\nGeneate: ", Program.Failure().reason.Str(), "\n");
    WGSL = std::move(Program.Get().wgsl);
    return WGSL;
}

int main(int argc, const char* argv[])
{
    try
    {
        auto SPIRV = ConvertHLSLtoSPIRV(HLSL::FillTextureCS);
        auto WGSL  = ConvertSPIRVtoWGSL(SPIRV);
        std::cout << WGSL << "\n";
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}
