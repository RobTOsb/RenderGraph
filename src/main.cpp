// Cornell box rendered through the render graph: reversed Z depth prepass, G-buffer, SSAO or RTAO, blur, deferred lighting.
// Left drag orbits, the wheel zooms, 1-4 pick lit/AO/albedo/normals, O toggles AO, R auto-rotates and Esc quits.

#include <DirectXMath.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "D3D12RenderContext.h"
#include "RenderContext.h"
#include "RenderGraph.h"
#include "Window.h"
#include "imgui.h"

using namespace DirectX;
using rg::Access;
using rg::Pass;
using rg::PassContext;
using rg::RenderGraph;

namespace
{
	constexpr uint32_t kFramesInFlight = 2;
	constexpr float kNearZ = 0.05f;
	constexpr float kFarZ = 100.0f;

	// Matches Vertex in scene.slang.
	struct Vertex
	{
		XMFLOAT3 position;
		XMFLOAT3 normal;
		XMFLOAT3 albedo;
		float emissive;
	};

	// Matches FrameConstants in common.slang.
	struct FrameConstants
	{
		XMFLOAT4X4 viewProj;
		XMFLOAT4X4 invViewProj;
		XMFLOAT3 cameraPos;
		float ssaoRadius;
		XMFLOAT3 lightPos;
		float lightIntensity;
		XMFLOAT2 size;
		float nearZ;
		float farZ;
		uint32_t debugView;
		uint32_t applyGamma;
		uint32_t useAO;
		float rtaoRadius;
	};
	static_assert(sizeof(FrameConstants) == 192);

	class SceneBuilder
	{
	public:
		// Corners in order around the face; the normal is given, so winding does not matter.
		void Quad(
				XMVECTOR a, XMVECTOR b, XMVECTOR c, XMVECTOR d, XMVECTOR normal, XMFLOAT3 albedo, float emissive = 0.0f)
		{
			for (XMVECTOR p: { a, b, c, a, c, d })
			{
				Vertex v;
				XMStoreFloat3(&v.position, p);
				XMStoreFloat3(&v.normal, normal);
				v.albedo = albedo;
				v.emissive = emissive;
				vertices_.push_back(v);
			}
		}

		// Box standing on the floor at (x, z), turned by yaw radians.
		void Box(float x, float z, float halfWidth, float height, float yaw, XMFLOAT3 albedo)
		{
			const XMMATRIX world = XMMatrixRotationY(yaw) * XMMatrixTranslation(x, 0.0f, z);
			auto corner = [&](float cx, float cy, float cz)
			{ return XMVector3TransformCoord(XMVectorSet(cx * halfWidth, cy * height, cz * halfWidth, 1.0f), world); };
			auto normal = [&](float nx, float ny, float nz)
			{ return XMVector3TransformNormal(XMVectorSet(nx, ny, nz, 0.0f), world); };

			Quad(corner(-1, 1, -1), corner(1, 1, -1), corner(1, 1, 1), corner(-1, 1, 1), normal(0, 1, 0), albedo);
			Quad(corner(-1, 0, -1), corner(1, 0, -1), corner(1, 1, -1), corner(-1, 1, -1), normal(0, 0, -1), albedo);
			Quad(corner(1, 0, 1), corner(-1, 0, 1), corner(-1, 1, 1), corner(1, 1, 1), normal(0, 0, 1), albedo);
			Quad(corner(-1, 0, 1), corner(-1, 0, -1), corner(-1, 1, -1), corner(-1, 1, 1), normal(-1, 0, 0), albedo);
			Quad(corner(1, 0, -1), corner(1, 0, 1), corner(1, 1, 1), corner(1, 1, -1), normal(1, 0, 0), albedo);
		}

		const std::vector<Vertex> &Vertices() const
		{
			return vertices_;
		}

	private:
		std::vector<Vertex> vertices_;
	};

	// Room from x -1..1, y 0..2, z -1..1, open towards -z where the camera starts.
	std::vector<Vertex> BuildCornellBox()
	{
		const XMFLOAT3 white = { 0.73f, 0.73f, 0.73f };
		const XMFLOAT3 red = { 0.63f, 0.065f, 0.05f };
		const XMFLOAT3 green = { 0.14f, 0.45f, 0.091f };
		auto p = [](float x, float y, float z) { return XMVectorSet(x, y, z, 1.0f); };
		auto n = [](float x, float y, float z) { return XMVectorSet(x, y, z, 0.0f); };

		SceneBuilder scene;
		scene.Quad(p(-1, 0, -1), p(1, 0, -1), p(1, 0, 1), p(-1, 0, 1), n(0, 1, 0), white); // floor
		scene.Quad(p(-1, 2, -1), p(-1, 2, 1), p(1, 2, 1), p(1, 2, -1), n(0, -1, 0), white); // ceiling
		scene.Quad(p(-1, 0, 1), p(1, 0, 1), p(1, 2, 1), p(-1, 2, 1), n(0, 0, -1), white); // back
		scene.Quad(p(-1, 0, -1), p(-1, 0, 1), p(-1, 2, 1), p(-1, 2, -1), n(1, 0, 0), red); // left
		scene.Quad(p(1, 0, 1), p(1, 0, -1), p(1, 2, -1), p(1, 2, 1), n(-1, 0, 0), green); // right
		const float l = 0.25f;
		scene.Quad(p(-l, 1.995f, -l),
				   p(-l, 1.995f, l),
				   p(l, 1.995f, l),
				   p(l, 1.995f, -l),
				   n(0, -1, 0),
				   { 1.0f, 0.95f, 0.85f },
				   1.0f); // light
		scene.Box(0.35f, -0.35f, 0.3f, 0.6f, -0.3f, white); // short box
		scene.Box(-0.35f, 0.3f, 0.3f, 1.2f, 0.3f, white); // tall box
		return scene.Vertices();
	}

	struct Camera
	{
		float yaw = 0.0f;
		float pitch = 0.1f;
		float distance = 4.2f;
		XMFLOAT3 target = { 0.0f, 1.0f, 0.0f };

		XMVECTOR Position() const
		{
			const XMVECTOR offset = XMVectorSet(std::sin(yaw) * std::cos(pitch),
												std::sin(pitch),
												-std::cos(yaw) * std::cos(pitch),
												0.0f);
			return XMVectorAdd(XMLoadFloat3(&target), XMVectorScale(offset, distance));
		}

		// Reversed Z: near and far are swapped, so depth is 1 at the near plane and 0 at the far plane.
		XMMATRIX ViewProj(float aspect) const
		{
			const XMMATRIX view = XMMatrixLookAtLH(Position(), XMLoadFloat3(&target), XMVectorSet(0, 1, 0, 0));
			const XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(40.0f), aspect, kFarZ, kNearZ);
			return view * proj;
		}
	};

	std::string ShaderPath(const char *name)
	{
		return std::string(RG_SANDBOX_SHADER_DIR) + "/" + name;
	}

	TextureDesc RenderTargetDesc(uint32_t width, uint32_t height, Format format, const char *name)
	{
		TextureDesc desc;
		desc.width = width;
		desc.height = height;
		desc.format = format;
		desc.clearValue = ClearValue{ { 0.0f, 0.0f, 0.0f, 0.0f } };
		desc.debugName = name;
		return desc;
	}

	// Binds the pipeline and frame constants and sets a full screen viewport.
	void BeginDraw(
			RenderContext &rc, PipelineHandle pipeline, ResourceHandle frameConstants, uint32_t width, uint32_t height)
	{
		rc.SetViewport(Viewport{ 0, 0, width, height, 0.0f, 1.0f });
		rc.SetScissorRect(ScissorRect{ 0, 0, width, height });
		rc.SetGraphicsPipeline(pipeline);
		rc.SetPrimitiveTopology(PrimitiveTopology::TriangleList);
		rc.SetConstantBufferByName("g_Frame", frameConstants);
	}

	PipelineHandle CreatePipeline(RenderContext &rc,
								  const char *shader,
								  std::vector<PipelineEntryPoint> entries,
								  const GraphicsPipelineCreateInfo &info)
	{
		return rc.CreateGraphicsPipeline(ShaderPath(shader), entries, info);
	}

	// Prints the warnings and errors the D3D12 debug layer recorded, which explain most failed calls.
	void PrintDebugLayerMessages(RenderContext &rc)
	{
		Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
		static_cast<D3D12RenderContext &>(rc).GetDevice()->GetNativeDevice().As(&infoQueue);
		for (UINT64 i = 0; infoQueue && i < infoQueue->GetNumStoredMessages(); ++i)
		{
			SIZE_T length = 0;
			infoQueue->GetMessage(i, nullptr, &length);
			std::vector<char> storage(length);
			auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
			infoQueue->GetMessage(i, message, &length);
			if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
			{
				std::fprintf(stderr, "  D3D12: %s\n", message->pDescription);
			}
		}
	}

	int Run()
	{
		WindowDesc windowDesc;
		windowDesc.title = "Render Graph Sandbox: Cornell box";
		windowDesc.width = 1280;
		windowDesc.height = 720;
		Window window(windowDesc);

		std::unique_ptr<RenderContext> rc(CreateRenderContext(&window, RenderAPI::D3D12, kFramesInFlight, 3));
		rc->CreatePerFrameResources();
		rc->InitialiseImgui(window.GetNativeWindow());

		const uint32_t width = window.GetWidth();
		const uint32_t height = window.GetHeight();
		const Format backBufferFormat = rc->GetBackBufferFormat();

		// Scene geometry, read by the vertex shader as a structured buffer.
		// It never changes after the upload, so the graph does not track it.
		const std::vector<Vertex> vertices = BuildCornellBox();
		const uint32_t vertexCount = static_cast<uint32_t>(vertices.size());
		BufferDesc vertexBufferDesc;
		vertexBufferDesc.sizeInBytes = sizeof(Vertex) * vertices.size();
		vertexBufferDesc.structureByteStride = sizeof(Vertex);
		vertexBufferDesc.debugName = "CornellBoxVertices";
		// Indices only feed the BLAS. The light is left out: it sits just under the ceiling and would occlude it.
		std::vector<uint32_t> indices;
		for (uint32_t i = 0; i < vertexCount; ++i)
		{
			if (vertices[i].emissive == 0.0f)
			{
				indices.push_back(i);
			}
		}
		BufferDesc indexBufferDesc;
		indexBufferDesc.sizeInBytes = sizeof(uint32_t) * indices.size();
		indexBufferDesc.structureByteStride = sizeof(uint32_t);
		indexBufferDesc.debugName = "CornellBoxIndices";
		const UploadBatchHandle upload = rc->BeginUpload();
		const ResourceHandle vertexBuffer =
				rc->CreateBuffer(vertexBufferDesc, ResourceViewType::ShaderResource, vertices.data(), upload);
		const ResourceHandle indexBuffer =
				rc->CreateBuffer(indexBufferDesc, ResourceViewType::ShaderResource, indices.data(), upload);
		rc->EndUpload(upload);

		// One BLAS and one instance for RTAO, built in the first frame. Like the vertices they never change,
		// so they stay out of the graph.
		BLASGeometry geometry;
		geometry.vertexBuffer = vertexBuffer;
		geometry.vertexCount = vertexCount;
		geometry.vertexStride = sizeof(Vertex);
		geometry.indexBuffer = indexBuffer;
		geometry.indexCount = static_cast<uint32_t>(indices.size());
		const AccelerationStructureHandle blas = rc->CreateBLAS(geometry);
		TLASInstance instance;
		instance.blas = blas;
		instance.transform[0][0] = instance.transform[1][1] = instance.transform[2][2] = 1.0f;
		const AccelerationStructureHandle tlas = rc->CreateTLAS(std::span<const TLASInstance>(&instance, 1));
		const ResourceHandle sceneSRV = rc->GetAccelerationStructureSRV(tlas);

		// One constant buffer per frame in flight, so the CPU never writes one the GPU is reading.
		std::array<ResourceHandle, kFramesInFlight> frameConstantBuffers;
		for (uint32_t i = 0; i < kFramesInFlight; ++i)
		{
			frameConstantBuffers[i] = rc->CreateConstantBuffer(sizeof(FrameConstants), "FrameConstants");
		}
		ResourceHandle frameConstants = frameConstantBuffers[0];

		// Pipelines. The prepass and G-buffer pass share a vertex shader so their depths match exactly.
		GraphicsPipelineCreateInfo depthInfo;
		depthInfo.rasterizerState.cullMode = D3D12_CULL_MODE_NONE;
		depthInfo.depthStencilState = DepthStencilStateDesc::ReversedZ();
		depthInfo.dsvFormat = DXGI_FORMAT_D32_FLOAT;
		depthInfo.rtvFormats = { DXGI_FORMAT_R32G32B32A32_FLOAT };
		const PipelineHandle depthPipeline = CreatePipeline(
				*rc,
				"scene.slang",
				{ { "vsMain", PipelineShaderStage::Vertex }, { "psPrepass", PipelineShaderStage::Pixel } },
				depthInfo);

		GraphicsPipelineCreateInfo gbufferInfo = depthInfo;
		gbufferInfo.depthStencilState.depthFunc = D3D12_COMPARISON_FUNC_EQUAL;
		gbufferInfo.depthStencilState.depthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
		gbufferInfo.rtvFormats = { DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM };
		const PipelineHandle gbufferPipeline = CreatePipeline(
				*rc,
				"scene.slang",
				{ { "vsMain", PipelineShaderStage::Vertex }, { "psGBuffer", PipelineShaderStage::Pixel } },
				gbufferInfo);

		GraphicsPipelineCreateInfo fullscreenInfo;
		fullscreenInfo.rasterizerState.cullMode = D3D12_CULL_MODE_NONE;
		fullscreenInfo.depthStencilState.depthEnable = FALSE;
		fullscreenInfo.depthStencilState.depthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
		const PipelineHandle ssaoPipeline = rc->CreateComputePipeline(ShaderPath("ssao.slang"), "csSSAO");
		const PipelineHandle blurPipeline = rc->CreateComputePipeline(ShaderPath("ssao.slang"), "csBlur");
		const PipelineHandle rtaoPipeline = rc->CreateComputePipeline(ShaderPath("rtao.slang"), "csRTAO");

		GraphicsPipelineCreateInfo lightingInfo = fullscreenInfo;
		lightingInfo.rtvFormats = { static_cast<DXGI_FORMAT>(backBufferFormat) };
		const PipelineHandle lightingPipeline = CreatePipeline(
				*rc,
				"lighting.slang",
				{ { "vsFullscreen", PipelineShaderStage::Vertex }, { "psLighting", PipelineShaderStage::Pixel } },
				lightingInfo);

		// The graph is built and compiled once; the back buffer import is rebound every frame.
		RenderGraph graph;
		TextureDesc depthDesc = RenderTargetDesc(width, height, Format::R32_TYPELESS, "Depth");
		depthDesc.clearValue.depth = 0.0f; // reversed Z clears to the far plane at 0
		const auto depth = graph.CreateTexture(depthDesc);
		const auto albedo = graph.CreateTexture(RenderTargetDesc(width, height, Format::R8G8B8A8_UNORM, "Albedo"));
		const auto normal = graph.CreateTexture(RenderTargetDesc(width, height, Format::R16G16B16A16_FLOAT, "Normal"));
		// Normal and depth for AO, written by the prepass so AO shares nothing with the G-buffer pass.
		const auto aoInput = graph.CreateTexture(RenderTargetDesc(width, height, Format::R32G32B32A32_FLOAT, "AOInput"));
		const auto aoRaw = graph.CreateTexture(RenderTargetDesc(width, height, Format::R8_UNORM, "AORaw"));
		const auto ao = graph.CreateTexture(RenderTargetDesc(width, height, Format::R8_UNORM, "AO"));
		const auto backBuffer = graph.ImportTexture(rc->GetCurrentBackBufferHandle(), "BackBuffer");

		// Section 1: the prepass. Section 2: the G-buffer on graphics while AO runs on async compute.
		// Section 3: lighting and the UI, once both are done.
		graph.AddPass(Pass::Graphics("DepthPrepass")
							  .Use(depth, Access::DepthStencilWrite)
							  .Writes(aoInput)
							  .Executes(
									  [&](PassContext &ctx)
									  {
										  const Resource depthTarget = ctx.GetResource(depth);
										  const Resource inputTarget = ctx.GetResource(aoInput);
										  float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
										  ctx.rc.SetRenderTargets(&inputTarget, 1, depthTarget);
										  ctx.rc.ClearRenderTarget(inputTarget, clear);
										  ctx.rc.ClearDepthStencil(depthTarget, 0.0f);
										  BeginDraw(ctx.rc, depthPipeline, frameConstants, width, height);
										  ctx.rc.SetSRVByName("g_Vertices", vertexBuffer);
										  ctx.rc.Draw(vertexCount);
									  }));

		graph.Submit();

		graph.AddPass(Pass::Graphics("GBuffer")
							  .Use(depth, Access::DepthStencilRead)
							  .Writes(albedo)
							  .Writes(normal)
							  .Executes(
									  [&](PassContext &ctx)
									  {
										  const Resource targets[] = { ctx.GetResource(normal),
																	   ctx.GetResource(albedo) };
										  float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
										  ctx.rc.ClearRenderTarget(targets[0], clear);
										  ctx.rc.ClearRenderTarget(targets[1], clear);
										  ctx.rc.SetRenderTargets(targets, 2, ctx.GetResource(depth), true);
										  BeginDraw(ctx.rc, gbufferPipeline, frameConstants, width, height);
										  ctx.rc.SetSRVByName("g_Vertices", vertexBuffer);
										  ctx.rc.Draw(vertexCount);
									  }));

		// SSAO and RTAO read the same input and write the same raw AO, so the checkbox picks one each frame
		// without recompiling the graph. Setting a pipeline resets its bindings, so everything is bound after it.
		bool useRTAO = true;
		auto dispatchAO = [&](PassContext &ctx, PipelineHandle pipeline, rg::ResourceId output, rg::ResourceId rawAO = {})
		{
			ctx.rc.SetComputePipeline(pipeline);
			ctx.rc.SetComputeConstantBufferByName("g_Frame", frameConstants);
			ctx.rc.SetComputeSRVByName("g_Input", ctx.Get(aoInput));
			if (pipeline == rtaoPipeline)
			{
				ctx.rc.SetComputeSRVByName("g_Scene", sceneSRV);
			}
			if (rawAO.IsValid())
			{
				ctx.rc.SetComputeSRVByName("g_AO", ctx.Get(rawAO));
			}
			ctx.rc.SetComputeUAVByName("g_Out", ctx.Get(output));
			ctx.rc.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
		};
		graph.AddPass(Pass::Compute("AO")
							  .OnQueue(QueueType::Compute)
							  .Reads(aoInput)
							  .Writes(aoRaw)
							  .Executes([&](PassContext &ctx)
										{ dispatchAO(ctx, useRTAO ? rtaoPipeline : ssaoPipeline, aoRaw); }));
		graph.AddPass(Pass::Compute("AOBlur")
							  .OnQueue(QueueType::Compute)
							  .Reads(aoInput)
							  .Reads(aoRaw)
							  .Writes(ao)
							  .Executes([&](PassContext &ctx) { dispatchAO(ctx, blurPipeline, ao, aoRaw); }));
		graph.Submit();

		graph.AddPass(Pass::Graphics("Lighting")
							  .Reads(albedo)
							  .Reads(normal)
							  .Reads(depth)
							  .Reads(ao)
							  .Writes(backBuffer)
							  .Executes(
									  [&](PassContext &ctx)
									  {
										  const Resource target = ctx.GetResource(backBuffer);
										  ctx.rc.SetRenderTargets(&target, 1);
										  BeginDraw(ctx.rc, lightingPipeline, frameConstants, width, height);
										  ctx.rc.SetSRVByName("g_Albedo", ctx.Get(albedo));
										  ctx.rc.SetSRVByName("g_Normal", ctx.Get(normal));
										  ctx.rc.SetSRVByName("g_Depth", ctx.Get(depth));
										  ctx.rc.SetSRVByName("g_AO", ctx.Get(ao));
										  ctx.rc.Draw(3);
									  }));
		graph.AddPass(Pass::Graphics("ImGui")
							  .Use(backBuffer, Access::ColourAttachment)
							  .Executes(
									  [&](PassContext &ctx)
									  {
										  const Resource target = ctx.GetResource(backBuffer);
										  ctx.rc.SetRenderTargets(&target, 1);
										  ctx.rc.ImguiRender();
									  }));

		if (!graph.Compile(*rc))
		{
			std::fprintf(stderr, "Render graph failed to compile: %s\n", graph.GetError().c_str());
			return 1;
		}
		
		std::printf("%s\n", graph.Dump().c_str());
		std::ofstream("RenderGraph.html", std::ios::binary) << graph.DumpHtml("Cornell box");

		Camera camera;
		FrameConstants constants = {};
		constants.ssaoRadius = 0.35f;
		constants.rtaoRadius = 0.5f;
		constants.lightPos = { 0.0f, 1.98f, 0.0f };
		constants.lightIntensity = 6.0f;
		constants.size = { static_cast<float>(width), static_cast<float>(height) };
		constants.nearZ = kNearZ;
		constants.farZ = kFarZ;
		constants.applyGamma = backBufferFormat != Format::R8G8B8A8_UNORM_SRGB ? 1u : 0u;
		constants.useAO = 1;

		uint64_t frame = 0;
		bool autoRotate = false;
		auto lastTime = std::chrono::steady_clock::now();
		try
		{
			while (window.ProcessEvents())
			{
				const float dx = window.GetMouseDeltaX();
				const float dy = window.GetMouseDeltaY();
				if (window.IsLeftMouseButtonDown())
				{
					camera.yaw -= dx * 0.005f;
					camera.pitch = std::clamp(camera.pitch + dy * 0.005f, -0.2f, 1.3f);
				}
				camera.distance = std::clamp(camera.distance - window.GetMouseWheelDelta() * 0.3f, 1.5f, 10.0f);
				for (uint32_t view = 0; view < 4; ++view)
				{
					if (window.WasKeyToggled('1' + view))
					{
						constants.debugView = view;
					}
				}
				if (window.WasKeyToggled('O'))
				{
					constants.useAO ^= 1u;
				}
				if (window.WasKeyToggled('R'))
				{
					autoRotate = !autoRotate;
				}
				const auto now = std::chrono::steady_clock::now();
				const float deltaSeconds = std::chrono::duration<float>(now - lastTime).count();
				lastTime = now;
				if (autoRotate)
				{
					camera.yaw += deltaSeconds * 0.4f;
				}

				const XMMATRIX viewProj = camera.ViewProj(static_cast<float>(width) / static_cast<float>(height));
				XMStoreFloat4x4(&constants.viewProj, viewProj);
				XMStoreFloat4x4(&constants.invViewProj, XMMatrixInverse(nullptr, viewProj));
				XMStoreFloat3(&constants.cameraPos, camera.Position());

				rc->ImguiNewFrame();
				ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
				ImGui::Begin("Ambient occlusion", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
				ImGui::Checkbox("Ray traced (RTAO)", &useRTAO);
				ImGui::TextUnformatted(useRTAO ? "8 rays per pixel against the scene TLAS" : "16 depth buffer samples per pixel");
				ImGui::End();

				// BeginFrame waits for this slot's previous frame, so its constant buffer is free to write.
				rc->BeginFrame();
				frameConstants = frameConstantBuffers[frame % kFramesInFlight];
				rc->UpdateConstantBuffer(frameConstants, &constants, sizeof(constants));
				rc->BeginGraphicsCommandList();
				rc->SetDescriptorHeaps();
				if (frame == 0)
				{
					// Recorded before the prepass, whose submit the compute queue waits on before tracing.
					rc->BuildAccelerationStructure(blas);
					rc->UAVBarrierForAccelerationStructures();
					rc->BuildAccelerationStructure(tlas);
					rc->UAVBarrierForAccelerationStructures();
				}

				const ResourceHandle currentBackBuffer = rc->GetCurrentBackBufferHandle();
				graph.RebindImport(backBuffer, currentBackBuffer);
				graph.Execute(*rc);
				rc->TransitionResource(currentBackBuffer,
									   ResourceLayout::Present,
									   ResourceSync::None,
									   ResourceAccess::NoAccess);

				rc->EndGraphicsCommandList();
				rc->EndFrame();
				rc->Present();
				++frame;
			}
		}
		catch (const std::exception &)
		{
			PrintDebugLayerMessages(*rc);
			throw;
		}

		rc->WaitForGPUIdle();
		PrintDebugLayerMessages(*rc);
		graph.ReleaseResources();
		rc->DestroyAccelerationStructure(tlas);
		rc->DestroyAccelerationStructure(blas);
		return 0;
	}
} // namespace

int main()
{
	try
	{
		return Run();
	}
	catch (const std::exception &e)
	{
		std::fprintf(stderr, "Fatal: %s\n", e.what());
		return 1;
	}
}
