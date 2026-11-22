#include "D3D12RenderContext.h"

#include <crtdbg.h>
#include <cstdlib>
#include <cstring>
#include <exception>

#include "TestCommon.h"

namespace rgtest
{
	std::vector<TestCase> &Registry()
	{
		static std::vector<TestCase> tests;
		return tests;
	}

	std::string ReadEnv(const char *name)
	{
		char *value = nullptr;
		size_t length = 0;
		if (_dupenv_s(&value, &length, name) != 0 || value == nullptr)
		{
			return {};
		}
		std::string result(value);
		std::free(value);
		return result;
	}

	bool UseWarp()
	{
		return ReadEnv("RG_TEST_WARP") == "1";
	}

	namespace
	{
		template<typename Options>
		Options QueryOptions(RenderContext &rc, D3D12_FEATURE feature)
		{
			Options options = {};
			auto &d3d12 = static_cast<D3D12RenderContext &>(rc);
			d3d12.GetDevice()->GetNativeDevice()->CheckFeatureSupport(feature, &options, sizeof(options));
			return options;
		}
	} // namespace

	bool SupportsMeshShaders(RenderContext &rc)
	{
		return QueryOptions<D3D12_FEATURE_DATA_D3D12_OPTIONS7>(rc, D3D12_FEATURE_D3D12_OPTIONS7).MeshShaderTier !=
			   D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
	}

	bool SupportsRaytracing(RenderContext &rc)
	{
		return QueryOptions<D3D12_FEATURE_DATA_D3D12_OPTIONS5>(rc, D3D12_FEATURE_D3D12_OPTIONS5).RaytracingTier >=
			   D3D12_RAYTRACING_TIER_1_1;
	}

	bool SupportsShadingRateImages(RenderContext &rc)
	{
		return QueryOptions<D3D12_FEATURE_DATA_D3D12_OPTIONS6>(rc, D3D12_FEATURE_D3D12_OPTIONS6).VariableShadingRateTier >=
			   D3D12_VARIABLE_SHADING_RATE_TIER_2;
	}

	namespace
	{
		struct Constants2
		{
			uint32_t a, b, c, d;
		};

		Microsoft::WRL::ComPtr<ID3D12InfoQueue> GetInfoQueue(RenderContext &rc)
		{
			Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue;
			auto &d3d12 = static_cast<D3D12RenderContext &>(rc);
			d3d12.GetDevice()->GetNativeDevice().As(&infoQueue);
			return infoQueue;
		}
	} // namespace

	GpuTest::GpuTest()
	{
		WindowDesc desc;
		desc.title = "RenderGraphTests";
		desc.width = 128;
		desc.height = 128;
		window_ = std::make_unique<Window>(desc);
		rc_.reset(CreateRenderContext(window_.get(), RenderAPI::D3D12, 2, 2, UseWarp()));
		assert(rc_ && "failed to create the render context");
		rc_->CreatePerFrameResources();

		if (auto infoQueue = GetInfoQueue(*rc_))
		{
			// Report errors through the stored-message check below instead of breaking into a debugger.
			infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
			infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);

			// Root SRV/UAV addresses in aliased transient memory always trigger the first; the tests check aliasing.
			// Tests with empty passes submit barrier-only lists, which the second is an optimisation hint about.
			D3D12_MESSAGE_ID denied[] = { D3D12_MESSAGE_ID_HEAP_ADDRESS_RANGE_INTERSECTS_MULTIPLE_BUFFERS,
										  D3D12_MESSAGE_ID_NON_OPTIMAL_BARRIER_ONLY_EXECUTE_COMMAND_LISTS };
			D3D12_INFO_QUEUE_FILTER filter = {};
			filter.DenyList.NumIDs = static_cast<UINT>(std::size(denied));
			filter.DenyList.pIDList = denied;
			infoQueue->AddStorageFilterEntries(&filter);
			infoQueue->ClearStoredMessages();
		}

		fill_ = LoadPipeline("fill.slang");
		add_ = LoadPipeline("add.slang");
		combine_ = LoadPipeline("combine.slang");
		increment_ = LoadPipeline("increment.slang");
		textureToBuffer_ = LoadPipeline("texture_to_buffer.slang");
	}

	GpuTest::~GpuTest()
	{
		rc_->WaitForGPUIdle();
		if (!debugLayerChecked_)
		{
			AssertNoDebugLayerErrors();
		}
	}

	PipelineHandle GpuTest::LoadPipeline(const char *file)
	{
		const std::string path = std::string(RG_TEST_SHADER_DIR) + "/" + file;
		const PipelineHandle handle = rc_->CreateComputePipeline(path, "main");
		if (handle == InvalidPipelineHandle)
		{
			std::fprintf(stderr, "failed to compile %s\n", path.c_str());
		}
		assert(handle != InvalidPipelineHandle);
		return handle;
	}

	void GpuTest::BeginFrame()
	{
		rc_->BeginFrame();
		rc_->BeginGraphicsCommandList();
		rc_->SetDescriptorHeaps();
	}

	void GpuTest::EndFrame(bool waitForGpu)
	{
		rc_->EndGraphicsCommandList();
		rc_->EndFrame();
		if (waitForGpu)
		{
			rc_->WaitForGPUIdle();
		}
	}

	ResourceHandle GpuTest::CreateBuffer(uint32_t elementCount, const char *name)
	{
		return rc_->CreateBuffer(UIntBuffer(elementCount, name), ResourceViewType::UnorderedAccess);
	}

	std::vector<uint32_t> GpuTest::Read(ResourceHandle buffer, uint32_t elementCount)
	{
		const uint64_t size = static_cast<uint64_t>(elementCount) * sizeof(uint32_t);
		BufferDesc desc;
		desc.sizeInBytes = size;
		desc.debugName = "TestReadback";
		const ResourceHandle readback = rc_->CreateReadbackBuffer(desc);

		BeginFrame();
		rc_->TransitionResource(buffer, ResourceLayout::Undefined, ResourceSync::Copy, ResourceAccess::CopySource);
		rc_->CopyBufferRegion(readback, 0, buffer, 0, size);
		EndFrame(true);

		std::vector<uint32_t> values(elementCount);
		rc_->ReadReadbackBuffer(readback, values.data(), 0, size);
		rc_->DestroyResource(readback);
		return values;
	}

	void GpuTest::Fill(rg::PassContext &ctx, rg::ResourceId output, uint32_t value, uint32_t count, uint32_t offset)
	{
		const Constants2 constants{ value, count, offset, 0 };
		ctx.rc.SetComputePipeline(fill_);
		ctx.rc.SetComputeUAVByName("Output", ctx.Get(output));
		ctx.rc.SetCompute32BitConstants(&constants, 4);
		ctx.rc.Dispatch((count + 63) / 64, 1, 1);
	}

	void GpuTest::Add(rg::PassContext &ctx,
					  rg::ResourceId input,
					  rg::ResourceId output,
					  uint32_t value,
					  uint32_t count,
					  uint32_t iterations)
	{
		const Constants2 constants{ value, count, iterations, 0 };
		ctx.rc.SetComputePipeline(add_);
		ctx.rc.SetComputeSRVByName("Input", ctx.Get(input));
		ctx.rc.SetComputeUAVByName("Output", ctx.Get(output));
		ctx.rc.SetCompute32BitConstants(&constants, 4);
		ctx.rc.Dispatch((count + 63) / 64, 1, 1);
	}

	void GpuTest::Combine(rg::PassContext &ctx, rg::ResourceId a, rg::ResourceId b, rg::ResourceId output, uint32_t count)
	{
		const Constants2 constants{ count, 0, 0, 0 };
		ctx.rc.SetComputePipeline(combine_);
		ctx.rc.SetComputeSRVByName("InputA", ctx.Get(a));
		ctx.rc.SetComputeSRVByName("InputB", ctx.Get(b));
		ctx.rc.SetComputeUAVByName("Output", ctx.Get(output));
		ctx.rc.SetCompute32BitConstants(&constants, 4);
		ctx.rc.Dispatch((count + 63) / 64, 1, 1);
	}

	void GpuTest::Increment(rg::PassContext &ctx, rg::ResourceId buffer, uint32_t value, uint32_t count)
	{
		const Constants2 constants{ value, count, 0, 0 };
		ctx.rc.SetComputePipeline(increment_);
		ctx.rc.SetComputeUAVByName("Data", ctx.Get(buffer));
		ctx.rc.SetCompute32BitConstants(&constants, 4);
		ctx.rc.Dispatch((count + 63) / 64, 1, 1);
	}

	void GpuTest::TextureToBuffer(rg::PassContext &ctx,
								  rg::ResourceId texture,
								  rg::ResourceId output,
								  uint32_t width,
								  uint32_t height)
	{
		const Constants2 constants{ width, height, 0, 0 };
		ctx.rc.SetComputePipeline(textureToBuffer_);
		ctx.rc.SetComputeSRVByName("InputTexture", ctx.Get(texture));
		ctx.rc.SetComputeUAVByName("Output", ctx.Get(output));
		ctx.rc.SetCompute32BitConstants(&constants, 4);
		ctx.rc.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
	}

	bool GpuTest::HasDebugLayerErrors()
	{
		auto infoQueue = GetInfoQueue(*rc_);
		for (UINT64 i = 0; infoQueue && i < infoQueue->GetNumStoredMessages(); ++i)
		{
			SIZE_T length = 0;
			infoQueue->GetMessage(i, nullptr, &length);
			std::vector<char> storage(length);
			auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
			infoQueue->GetMessage(i, message, &length);
			if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR || message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)
			{
				return true;
			}
		}
		return false;
	}

	void GpuTest::AssertNoDebugLayerErrors()
	{
		debugLayerChecked_ = true;
		auto infoQueue = GetInfoQueue(*rc_);
		if (!infoQueue)
		{
			std::printf("  (D3D12 debug layer not available: validation skipped)\n");
			return;
		}

		uint32_t errors = 0;
		const UINT64 count = infoQueue->GetNumStoredMessages();
		for (UINT64 i = 0; i < count; ++i)
		{
			SIZE_T length = 0;
			infoQueue->GetMessage(i, nullptr, &length);
			std::vector<char> storage(length);
			auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
			infoQueue->GetMessage(i, message, &length);
			const bool isError = message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
								 message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION;
			if (isError || message->Severity == D3D12_MESSAGE_SEVERITY_WARNING)
			{
				std::fprintf(stderr,
							 "  D3D12 %s %d: %s\n",
							 isError ? "ERROR" : "warning",
							 static_cast<int>(message->ID),
							 message->pDescription);
			}
			errors += isError ? 1 : 0;
		}
		infoQueue->ClearStoredMessages();
		assert(errors == 0 && "the D3D12 debug layer reported errors");
	}

	void AssertBufferEquals(const std::vector<uint32_t> &actual,
							const std::function<uint32_t(uint32_t)> &expected,
							const char *what,
							const rg::RenderGraph *graph)
	{
		for (uint32_t i = 0; i < actual.size(); ++i)
		{
			if (actual[i] != expected(i))
			{
				std::fprintf(stderr,
							 "%s: element %u is %u, expected %u\n",
							 what,
							 i,
							 actual[i],
							 expected(i));
				if (graph)
				{
					std::fprintf(stderr, "%s\n", graph->Dump().c_str());
				}
				assert(false && "buffer contents differ from the expected values");
			}
		}
	}

	int FindEvent(const rg::RenderGraph &graph, const std::function<bool(const rg::ExecutionEvent &)> &predicate)
	{
		const auto &trace = graph.GetExecutionTrace();
		for (size_t i = 0; i < trace.size(); ++i)
		{
			if (predicate(trace[i]))
			{
				return static_cast<int>(i);
			}
		}
		return -1;
	}

	int FindPassEvent(const rg::RenderGraph &graph, rg::PassId pass)
	{
		return FindEvent(graph,
						 [pass](const rg::ExecutionEvent &e)
						 { return e.type == rg::ExecutionEventType::Pass && e.pass == pass; });
	}

	const rg::BarrierInfo *FindBarrier(const rg::CompiledBatch &batch, rg::ResourceId resource)
	{
		for (const rg::BarrierInfo &barrier: batch.barriers)
		{
			if (barrier.resource == resource)
			{
				return &barrier;
			}
		}
		return nullptr;
	}

	const rg::BarrierInfo *FindRelease(const rg::RenderGraph &graph, uint32_t section, QueueType queue, rg::ResourceId resource)
	{
		for (const rg::BarrierInfo &barrier: graph.GetQueueReleases(section, queue))
		{
			if (barrier.resource == resource)
			{
				return &barrier;
			}
		}
		return nullptr;
	}
} // namespace rgtest

int main(int argc, char **argv)
{
	// Failed asserts print to stderr and abort instead of opening a dialog, so CTest sees the failure.
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
	_set_error_mode(_OUT_TO_STDERR);
	SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

	const auto &tests = rgtest::Registry();
	if (argc > 1 && std::strcmp(argv[1], "--list") == 0)
	{
		for (const auto &test: tests)
		{
			std::printf("%s\n", test.name);
		}
		return 0;
	}

	int run = 0;
	for (const auto &test: tests)
	{
		if (argc > 1 && std::strcmp(argv[1], test.name) != 0)
		{
			continue;
		}
		std::printf("[ RUN  ] %s\n", test.name);
		std::fflush(stdout);
		try
		{
			test.fn();
		}
		catch (const std::exception &e)
		{
			std::fprintf(stderr, "[ FAIL ] %s: exception: %s\n", test.name, e.what());
			return 1;
		}
		std::printf("[ OK   ] %s\n", test.name);
		++run;
	}

	if (run == 0)
	{
		std::fprintf(stderr, "no test named '%s'\n", argc > 1 ? argv[1] : "");
		return 1;
	}
	std::printf("%d test(s) passed\n", run);
	return 0;
}
