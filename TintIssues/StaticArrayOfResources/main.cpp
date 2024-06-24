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

const std::string TestCS = R"(

RWTexture2D<float4> Tex2D_F2[2];

[numthreads(2,4,8)]
void main(uint3 Gid : SV_GroupID,
            uint3 GTid : SV_GroupThreadID)
{
    Tex2D_F2[0][GTid.xy] = float4(0.0, 0.0, 0.0, 1.0); 
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
    Shader.setAutoMapBindings(true);
    Shader.setAutoMapLocations(true);

    TBuiltInResource Resources;
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

    std::string   WGSL;
    tint::Program Program = tint::spirv::reader::Read(SPIRV, {true});

    if (!Program.IsValid())
        LOG_ERROR_AND_THROW("Tint SPIR-V reader failure:\nParser: ", Program.Diagnostics().Str(), "\n");

    auto GenerationResult = tint::wgsl::writer::Generate(Program, {});

    if (GenerationResult != tint::Success)
        LOG_ERROR_AND_THROW("Tint WGSL writer failure:\nGeneate: ", GenerationResult.Failure().reason.Str(), "\n");
    WGSL = std::move(GenerationResult->wgsl);
    return WGSL;
}

int main(int argc, const char* argv[])
{
    try
    {
        auto SPIRV = ConvertHLSLtoSPIRV(HLSL::TestCS);
        auto WGSL  = ConvertSPIRVtoWGSL(SPIRV);
        std::cout << WGSL << "\n";
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}
