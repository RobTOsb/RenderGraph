// Cornell box rendered through the render graph: reversed Z depth prepass, G-buffer, SSAO or RTAO, blur, deferred
// lighting. Left drag orbits, the wheel zooms, 1-4 pick lit/AO/albedo/normals, O toggles AO, R auto-rotates and Esc
// quits.

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
#include <stdexcept>
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

	// The profile scope around the whole graph, on the graphics queue.
	constexpr const char *kGpuFrameScope = "GPU frame";

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
		windowDesc.fullscreen = true;
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

		// One BLAS and one instance for RTAO. The graph builds the BLAS once, in the first RTAO frame, and the TLAS
		// every frame into a transient buffer, so the TLAS object only holds the instances.
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
		const TLASBuildSizes tlasSizes = rc->GetTLASBuildSizes(tlas);

		// One constant buffer per frame in flight, so the CPU never writes one the GPU is reading.
		std::array<ResourceHandle, kFramesInFlight> frameConstantBuffers;
		for (uint32_t i = 0; i < kFramesInFlight; ++i)
		{
			frameConstantBuffers[i] = rc->CreateConstantBuffer(sizeof(FrameConstants), "FrameConstants");
		}
		ResourceHandle frameConstants = frameConstantBuffers[0];

		// Pipelines. The prepass and G-buffer pass share a vertex shader so their depths match exactly.
		GraphicsPipelineCreateInfo depthInfo;
		depthInfo.rasterizerState.cullMode = CullMode::None;
		depthInfo.depthStencilState = DepthStencilStateDesc::ReversedZ();
		depthInfo.depthStencilFormat = Format::D32_FLOAT;
		depthInfo.renderTargetFormats = { Format::R16G16B16A16_FLOAT };
		const PipelineHandle depthPipeline = CreatePipeline(
				*rc,
				"scene.slang",
				{ { "vsMain", PipelineShaderStage::Vertex }, { "psPrepass", PipelineShaderStage::Pixel } },
				depthInfo);

		// RTAO traces its own surfaces, so its prepass only writes depth.
		GraphicsPipelineCreateInfo depthOnlyInfo = depthInfo;
		depthOnlyInfo.renderTargetFormats = {};
		const PipelineHandle depthOnlyPipeline =
				CreatePipeline(*rc, "scene.slang", { { "vsMain", PipelineShaderStage::Vertex } }, depthOnlyInfo);

		GraphicsPipelineCreateInfo gbufferInfo = depthInfo;
		gbufferInfo.depthStencilState.depthFunc = CompareFunc::Equal;
		gbufferInfo.depthStencilState.depthWrite = false;
		gbufferInfo.renderTargetFormats = { Format::R16G16B16A16_FLOAT, Format::R8G8B8A8_UNORM };
		const PipelineHandle gbufferPipeline = CreatePipeline(
				*rc,
				"scene.slang",
				{ { "vsMain", PipelineShaderStage::Vertex }, { "psGBuffer", PipelineShaderStage::Pixel } },
				gbufferInfo);

		GraphicsPipelineCreateInfo fullscreenInfo;
		fullscreenInfo.rasterizerState.cullMode = CullMode::None;
		fullscreenInfo.depthStencilState.depthEnable = false;
		fullscreenInfo.depthStencilState.depthWrite = false;
		const PipelineHandle ssaoPipeline = rc->CreateComputePipeline(ShaderPath("ssao.slang"), "csSSAO");
		const PipelineHandle blurPipeline = rc->CreateComputePipeline(ShaderPath("ssao.slang"), "csBlur");
		const PipelineHandle rtaoPipeline = rc->CreateComputePipeline(ShaderPath("rtao.slang"), "csRTAO");

		GraphicsPipelineCreateInfo lightingInfo = fullscreenInfo;
		lightingInfo.renderTargetFormats = { backBufferFormat };
		const PipelineHandle lightingPipeline = CreatePipeline(
				*rc,
				"lighting.slang",
				{ { "vsFullscreen", PipelineShaderStage::Vertex }, { "psLighting", PipelineShaderStage::Pixel } },
				lightingInfo);

		// The graph is reset, recorded, compiled and executed every frame, so its passes can follow the UI:
		// the TLAS build and read are only in it while RTAO is on, and the BLAS build until it has run once.
		// Compile only does the work when the recording differs from the last one; other frames reuse it.
		RenderGraph graph;
		graph.SetProfilePasses(true); // each pass gets a GPU timestamp scope, shown in the UI
		bool useRTAO = true;
		bool asyncCompute = false; // off puts the AO and acceleration structure passes on the graphics queue
		bool blasBuilt = false;
		float compileMilliseconds = 0.0f;
		uint32_t fullCompiles = 0;
		auto executeGraph = [&](ResourceHandle currentBackBuffer)
		{
			graph.Reset();
			TextureDesc depthDesc = RenderTargetDesc(width, height, Format::R32_TYPELESS, "Depth");
			depthDesc.clearValue.depth = 0.0f; // reversed Z clears to the far plane at 0
			const auto depth = graph.CreateTexture(depthDesc);
			const auto albedo = graph.CreateTexture(RenderTargetDesc(width, height, Format::R8G8B8A8_UNORM, "Albedo"));
			const auto normal =
					graph.CreateTexture(RenderTargetDesc(width, height, Format::R16G16B16A16_FLOAT, "Normal"));
			// Normal and depth for AO and its blur: written by the prepass for SSAO, and by RTAO's primary rays.
			const auto aoInput = graph.CreateTexture(
					RenderTargetDesc(width, height, Format::R16G16B16A16_FLOAT, useRTAO ? "TracedSurface" : "AOInput"));
			const auto aoRaw = graph.CreateTexture(RenderTargetDesc(width, height, Format::R8_UNORM, "AORaw"));
			const auto ao = graph.CreateTexture(RenderTargetDesc(width, height, Format::R8_UNORM, "AO"));
			const auto backBuffer = graph.ImportTexture(currentBackBuffer, "BackBuffer");
			const QueueType aoQueue = asyncCompute ? QueueType::Compute : QueueType::Graphics;

			// The sections below are laid out for async compute. Without it the aoQueue passes run on graphics in
			// the same sections, and the graph orders them by their dependencies alone.
			// SSAO reads the prepass. Section 1: the prepass. Section 2: the G-buffer on graphics while SSAO runs on
			// async compute. Section 3: lighting and the UI.
			// RTAO traces its own surfaces, so AO needs nothing from graphics. Section 1: the prepass and G-buffer on
			// graphics while async compute builds the TLAS (after the BLAS in the first frame), traces and blurs.
			// Section 2: lighting and the UI.
			rg::ResourceId sceneTLAS;
			const bool buildBLAS = useRTAO && !blasBuilt;
			if (useRTAO)
			{
				// The BLAS keeps its contents, so it is imported. The TLAS and its scratch live for one execution.
				const auto sceneBLAS = graph.ImportBuffer(rc->GetAccelerationStructureResource(blas), "SceneBLAS");
				BufferDesc tlasDesc;
				tlasDesc.sizeInBytes = tlasSizes.resultBytes;
				tlasDesc.debugName = "SceneTLAS";
				sceneTLAS = graph.CreateBuffer(tlasDesc, ResourceViewType::ShaderResource);
				BufferDesc scratchDesc;
				scratchDesc.sizeInBytes = tlasSizes.scratchBytes;
				scratchDesc.structureByteStride = sizeof(uint32_t);
				scratchDesc.debugName = "TLASScratch";
				const auto tlasScratch = graph.CreateBuffer(scratchDesc);
				if (buildBLAS)
				{
					graph.AddPass(
							Pass::Compute("BuildBLAS")
									.OnQueue(aoQueue)
									.Use(sceneBLAS, Access::AccelerationStructureBuildWrite)
									.Executes([&](PassContext &ctx) { ctx.rc.BuildAccelerationStructure(blas); }));
				}
				graph.AddPass(
						Pass::Compute("BuildTLAS")
								.OnQueue(aoQueue)
								.Use(sceneBLAS, Access::AccelerationStructureBuildRead)
								.Use(tlasScratch, Access::AccelerationStructureScratch)
								.Use(sceneTLAS, Access::AccelerationStructureBuildWrite)
								// By value: the scratch id is scoped to this block, and the pass runs after it ends.
								.Executes([&, tlasScratch](PassContext &ctx)
										  { ctx.rc.BuildTLASInto(tlas, ctx.Get(sceneTLAS), ctx.Get(tlasScratch)); }));
			}
			Pass prepass = Pass::Graphics("DepthPrepass")
								   .Use(depth, Access::DepthStencilWrite)
								   .Executes(
										   [&](PassContext &ctx)
										   {
											   const Resource depthTarget = ctx.GetResource(depth);
											   if (useRTAO)
											   {
												   ctx.rc.SetRenderTargets(nullptr, 0, depthTarget);
											   }
											   else
											   {
												   const Resource inputTarget = ctx.GetResource(aoInput);
												   float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
												   ctx.rc.SetRenderTargets(&inputTarget, 1, depthTarget);
												   ctx.rc.ClearRenderTarget(inputTarget, clear);
											   }
											   ctx.rc.ClearDepthStencil(depthTarget, 0.0f);
											   BeginDraw(ctx.rc,
														 useRTAO ? depthOnlyPipeline : depthPipeline,
														 frameConstants,
														 width,
														 height);
											   ctx.rc.SetSRVByName("g_Vertices", vertexBuffer);
											   ctx.rc.Draw(vertexCount);
										   });
			if (!useRTAO)
			{
				prepass.Writes(aoInput);
			}
			graph.AddPass(std::move(prepass));
			if (!useRTAO)
			{
				graph.Submit();
			}

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

			// SSAO and the blur read the surface in aoInput, which RTAO writes. Setting a pipeline resets its
			// bindings, so everything is bound after it.
			auto dispatchAO =
					[&](PassContext &ctx, PipelineHandle pipeline, rg::ResourceId output, rg::ResourceId rawAO = {})
			{
				ctx.rc.SetComputePipeline(pipeline);
				ctx.rc.SetComputeConstantBufferByName("g_Frame", frameConstants);
				ctx.rc.SetComputeSRVByName("g_Input", ctx.Get(aoInput));
				if (rawAO.IsValid())
				{
					ctx.rc.SetComputeSRVByName("g_AO", ctx.Get(rawAO));
				}
				ctx.rc.SetComputeUAVByName("g_Out", ctx.Get(output));
				ctx.rc.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
			};
			if (useRTAO)
			{
				// The vertices and indices never change, so they stay out of the graph like in the raster passes.
				graph.AddPass(Pass::Compute("RTAO")
									  .OnQueue(aoQueue)
									  .Use(sceneTLAS, Access::AccelerationStructureRead)
									  .Writes(aoInput)
									  .Writes(aoRaw)
									  .Executes(
											  [&](PassContext &ctx)
											  {
												  ctx.rc.SetComputePipeline(rtaoPipeline);
												  ctx.rc.SetComputeConstantBufferByName("g_Frame", frameConstants);
												  ctx.rc.SetComputeSRVByName("g_Vertices", vertexBuffer);
												  ctx.rc.SetComputeSRVByName("g_Indices", indexBuffer);
												  ctx.rc.SetComputeSRVByName("g_Scene", ctx.Get(sceneTLAS));
												  ctx.rc.SetComputeUAVByName("g_Surface", ctx.Get(aoInput));
												  ctx.rc.SetComputeUAVByName("g_Out", ctx.Get(aoRaw));
												  ctx.rc.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
											  }));
			}
			else
			{
				graph.AddPass(Pass::Compute("SSAO").OnQueue(aoQueue).Reads(aoInput).Writes(aoRaw).Executes(
						[&](PassContext &ctx) { dispatchAO(ctx, ssaoPipeline, aoRaw); }));
			}
			graph.AddPass(Pass::Compute("AOBlur").OnQueue(aoQueue).Reads(aoInput).Reads(aoRaw).Writes(ao).Executes(
					[&](PassContext &ctx) { dispatchAO(ctx, blurPipeline, ao, aoRaw); }));
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

			const auto compileStart = std::chrono::steady_clock::now();
			if (!graph.Compile(*rc))
			{
				throw std::runtime_error("Render graph failed to compile: " + graph.GetError());
			}
			compileMilliseconds =
					std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - compileStart).count();
			// A full compile means the graph changed shape, so print it and save it as HTML.
			if (!graph.WasLastCompileReused())
			{
				++fullCompiles;
				std::printf("%s\n", graph.Dump().c_str());
				std::ofstream("RenderGraph.html", std::ios::binary) << graph.DumpHtml("Cornell box");
			}
			// The pass callbacks capture this scope's resources, so the graph executes before it returns.
			graph.Execute(*rc);
			blasBuilt = blasBuilt || buildBLAS;
		};

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
		// Frame time and FPS over half second windows, so the numbers are readable; the worst frame shows hitches.
		float windowSeconds = 0.0f;
		float windowWorstSeconds = 0.0f;
		uint32_t windowFrames = 0;
		float frameMilliseconds = 0.0f;
		float worstFrameMilliseconds = 0.0f;
		float framesPerSecond = 0.0f;
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
				windowSeconds += deltaSeconds;
				windowWorstSeconds = std::max(windowWorstSeconds, deltaSeconds);
				++windowFrames;
				if (windowSeconds >= 0.5f)
				{
					frameMilliseconds = windowSeconds * 1000.0f / static_cast<float>(windowFrames);
					worstFrameMilliseconds = windowWorstSeconds * 1000.0f;
					framesPerSecond = static_cast<float>(windowFrames) / windowSeconds;
					windowSeconds = 0.0f;
					windowWorstSeconds = 0.0f;
					windowFrames = 0;
				}
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
				ImGui::Text("%.1f fps, %.3f ms a frame (worst %.3f ms)",
							framesPerSecond,
							frameMilliseconds,
							worstFrameMilliseconds);
				// The graphics queue from the first pass to the last, waits for async compute included.
				const std::optional<GPUProfileScopeStats> gpuFrame = rc->GetProfileScopeStats(kGpuFrameScope);
				ImGui::Text("GPU frame %.3f ms (worst %.3f ms)",
							gpuFrame ? gpuFrame->averageMs : 0.0f,
							gpuFrame ? gpuFrame->maxMs : 0.0f);
				ImGui::Separator();
				ImGui::Checkbox("Ray traced (RTAO)", &useRTAO);
				ImGui::TextUnformatted(useRTAO ? "8 rays per pixel against the scene TLAS"
											   : "16 depth buffer samples per pixel");
				ImGui::Checkbox("Async compute", &asyncCompute);
				ImGui::TextUnformatted(asyncCompute ? "AO and the TLAS build overlap the raster passes"
													: "Every pass runs on the graphics queue");
				ImGui::Separator();
				ImGui::Text("Render graph: %u batches, %s in %.3f ms",
							graph.GetTotalBatchCount(),
							graph.WasLastCompileReused() ? "reused" : "compiled",
							compileMilliseconds);
				ImGui::Text("Full compiles so far: %u", fullCompiles);

				// GPU time of each pass, averaged over recent frames. Async compute overlaps graphics, so the
				// queue totals are busy time, not frame time.
				ImGui::Separator();
				const std::vector<rg::PassTiming> timings = graph.GetPassTimings(*rc);
				float queueTotals[2] = {};
				if (ImGui::BeginTable("GPU timings", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit))
				{
					ImGui::TableSetupColumn("Pass");
					ImGui::TableSetupColumn("Queue");
					ImGui::TableSetupColumn("Avg ms");
					ImGui::TableSetupColumn("Max ms");
					ImGui::TableHeadersRow();
					for (const rg::PassTiming &timing: timings)
					{
						ImGui::TableNextRow();
						ImGui::TableNextColumn();
						ImGui::TextUnformatted(timing.name.c_str());
						ImGui::TableNextColumn();
						ImGui::TextUnformatted(timing.queue == QueueType::Compute ? "Compute" : "Graphics");
						ImGui::TableNextColumn();
						if (timing.stats.sampleCount == 0)
						{
							ImGui::TextUnformatted("-");
							ImGui::TableNextColumn();
							ImGui::TextUnformatted("-");
							continue;
						}
						ImGui::Text("%.3f", timing.stats.averageMs);
						ImGui::TableNextColumn();
						ImGui::Text("%.3f", timing.stats.maxMs);
						queueTotals[timing.queue == QueueType::Compute ? 1 : 0] += timing.stats.averageMs;
					}
					ImGui::EndTable();
				}
				ImGui::Text("Graphics busy %.3f ms, compute busy %.3f ms", queueTotals[0], queueTotals[1]);
				ImGui::End();

				// BeginFrame waits for this slot's previous frame, so its constant buffer is free to write.
				rc->BeginFrame();
				frameConstants = frameConstantBuffers[frame % kFramesInFlight];
				rc->UpdateConstantBuffer(frameConstants, &constants, sizeof(constants));
				rc->BeginGraphicsCommandList();
				rc->SetDescriptorHeaps();

				const ResourceHandle currentBackBuffer = rc->GetCurrentBackBufferHandle();
				rc->BeginProfileScope(kGpuFrameScope);
				executeGraph(currentBackBuffer);
				rc->EndProfileScope();
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
