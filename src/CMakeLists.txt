add_library(RenderGraph STATIC
    RenderGraph.cpp
    RenderGraphHtml.cpp
)

target_include_directories(RenderGraph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# RenderContext.h includes wrapper (D3D12Pipeline.h) and DirectXTex headers, so consumers need them too.
target_link_libraries(RenderGraph PUBLIC D3D12RenderContext D3D12 DirectXTex)
# RenderContext.h pulls in windows.h; keep its min/max macros away from std::min/std::max.
target_compile_definitions(RenderGraph PUBLIC NOMINMAX)

add_executable(RenderGraphSandbox main.cpp)
target_link_libraries(RenderGraphSandbox PRIVATE RenderGraph)
target_compile_definitions(RenderGraphSandbox PRIVATE RG_SANDBOX_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
d3d12_deploy_runtime_dlls(RenderGraphSandbox)
