#pragma once

// Test fixture on the real D3D12 render context: asserts on the compiled graph, GPU results and debug layer.
// Include this header last: it re-enables assert() even in release builds.

#include "RenderContext.h"
#include "RenderGraph.h"
#include "Window.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rgtest
{
	using TestFn = void (*)();

	struct TestCase
	{
		const char *name;
		TestFn fn;
	};

	std::vector<TestCase> &Registry();

	// Environment variable value, or empty when unset.
	std::string ReadEnv(const char *name);
	bool UseWarp();

	// Optional device features; tests that need one skip when the device lacks it.
	bool SupportsMeshShaders(RenderContext &rc);
	bool SupportsRaytracing(RenderContext &rc); // DXR 1.1, which inline ray queries need
	bool SupportsShadingRateImages(RenderContext &rc);

	struct Registrar
	{
		Registrar(const char *name, TestFn fn)
		{
			Registry().push_back({ name, fn });
		}
	};

#define RG_TEST(name)                                                                                                  \
	static void name();                                                                                                \
	static const ::rgtest::Registrar name##_registrar(#name, &name);                                                   \
	static void name()

	// Same arithmetic as add.slang's busy loop (which contributes nothing when it does not run).
	inline uint32_t LcgLowBits(uint32_t index, uint32_t iterations)
	{
		if (iterations == 0)
		{
			return 0;
		}
		uint32_t h = index;
		for (uint32_t k = 0; k < iterations; ++k)
		{
			h = h * 1664525u + 1013904223u;
		}
		return h & 0xFu;
	}

	// Window + render context + the pipelines the tests dispatch.
	class GpuTest
	{
	public:
		GpuTest();
		~GpuTest();

		RenderContext &rc()
		{
			return *rc_;
		}

		// BeginFrame + open graphics list + descriptor heaps, ready for RenderGraph::Execute.
		void BeginFrame();
		// Closes and submits the frame; by default waits for the GPU so results can be read.
		void EndFrame(bool waitForGpu = true);

		ResourceHandle CreateBuffer(uint32_t elementCount, const char *name);
		// Copies a buffer into a readback buffer in its own frame and returns its contents.
		std::vector<uint32_t> Read(ResourceHandle buffer, uint32_t elementCount);

		// Dispatch helpers for pass callbacks.
		void Fill(rg::PassContext &ctx, rg::ResourceId output, uint32_t value, uint32_t count, uint32_t offset = 0);
		void Add(rg::PassContext &ctx,
				 rg::ResourceId input,
				 rg::ResourceId output,
				 uint32_t value,
				 uint32_t count,
				 uint32_t iterations = 0);
		void Combine(rg::PassContext &ctx, rg::ResourceId a, rg::ResourceId b, rg::ResourceId output, uint32_t count);
		void Increment(rg::PassContext &ctx, rg::ResourceId buffer, uint32_t value, uint32_t count);
		void TextureToBuffer(rg::PassContext &ctx, rg::ResourceId texture, rg::ResourceId output, uint32_t width, uint32_t height);

		// Asserts the debug layer stored no errors since the last call (or since creation).
		void AssertNoDebugLayerErrors();
		// Whether the debug layer stored an error since the last assert, without clearing anything.
		bool HasDebugLayerErrors();

	private:
		PipelineHandle LoadPipeline(const char *file);

		std::unique_ptr<Window> window_;
		std::unique_ptr<RenderContext> rc_;
		PipelineHandle fill_ = InvalidPipelineHandle;
		PipelineHandle add_ = InvalidPipelineHandle;
		PipelineHandle combine_ = InvalidPipelineHandle;
		PipelineHandle increment_ = InvalidPipelineHandle;
		PipelineHandle textureToBuffer_ = InvalidPipelineHandle;
		bool debugLayerChecked_ = false;
	};

	inline BufferDesc UIntBuffer(uint32_t count, const char *name)
	{
		BufferDesc desc;
		desc.sizeInBytes = static_cast<uint64_t>(count) * sizeof(uint32_t);
		desc.structureByteStride = sizeof(uint32_t);
		desc.usageFlags = ResourceUsageFlags::UnorderedAccess;
		desc.debugName = name;
		return desc;
	}

	// Prints the first mismatch (and the graph) before asserting, so a failure says what went wrong.
	void AssertBufferEquals(const std::vector<uint32_t> &actual,
							const std::function<uint32_t(uint32_t)> &expected,
							const char *what,
							const rg::RenderGraph *graph = nullptr);

	// Index of the first trace event matching the predicate, or -1.
	int FindEvent(const rg::RenderGraph &graph, const std::function<bool(const rg::ExecutionEvent &)> &predicate);
	int FindPassEvent(const rg::RenderGraph &graph, rg::PassId pass);

	// The barrier on resource recorded before the given batch, or nullptr.
	const rg::BarrierInfo *FindBarrier(const rg::CompiledBatch &batch, rg::ResourceId resource);
	const rg::BarrierInfo *FindRelease(const rg::RenderGraph &graph, uint32_t section, QueueType queue, rg::ResourceId resource);
} // namespace rgtest

// assert() stays active in every build configuration for the tests.
#undef NDEBUG
#include <cassert>
