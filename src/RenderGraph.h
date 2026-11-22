#pragma once

// Render graph after https://ipotrick.github.io/site/blogs/practical-efficient-rendergraphs/
// Record passes, Compile() schedules batches, barriers, queue syncs and aliasing, Execute() replays it.

#include "RenderContext.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Operators for the RenderContext's barrier types (global, like the types, so lookup finds them)

constexpr ResourceSync operator|(ResourceSync a, ResourceSync b)
{
	return static_cast<ResourceSync>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

constexpr ResourceAccess operator|(ResourceAccess a, ResourceAccess b)
{
	return static_cast<ResourceAccess>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

template<typename Flags>
bool HasAll(Flags value, Flags bits)
{
	return (static_cast<uint32_t>(value) & static_cast<uint32_t>(bits)) == static_cast<uint32_t>(bits);
}

inline bool operator==(const ResourceState &a, const ResourceState &b)
{
	return a.layout == b.layout && a.sync == b.sync && a.access == b.access;
}

namespace rg
{
	// Each access maps to the sync scope, access bits and texture layout a barrier needs.
	// Shader writes are UAV accesses, which include reads. Amplification and mesh shaders sync as vertex shading.
	enum class Access : uint8_t
	{
		// Compute shaders
		ComputeShaderRead, // SRV
		ComputeShaderWrite, // UAV
		ComputeShaderReadWrite, // UAV read-modify-write
		ClearUnorderedAccess, // ClearUnorderedAccessView*

		// Graphics shaders
		VertexShaderRead,
		VertexShaderWrite,
		MeshShaderRead, // amplification and mesh shaders
		MeshShaderWrite,
		PixelShaderRead,
		PixelShaderWrite,
		GraphicsShaderRead, // SRV in more than one graphics stage
		GraphicsShaderWrite, // UAV in more than one graphics stage
		ConstantBufferRead, // CBV in any shader stage, buffers only

		// Fixed function
		IndirectArgumentRead, // ExecuteIndirect arguments and counts, buffers only
		PredicationRead, // SetPredication, buffers only
		ShadingRateSource, // variable rate shading image, textures only
		CopySource,
		CopyDest,
		ColourAttachment, // textures only
		DepthStencilWrite, // textures only
		DepthStencilRead, // textures only

		// Ray tracing. Acceleration structures come from the RenderContext, so they are always imported buffers.
		RaytracingShaderRead, // SRV in DispatchRays shaders
		RaytracingShaderWrite, // UAV in DispatchRays shaders
		AccelerationStructureRead, // TraceRay, or inline ray queries in any shader stage
		AccelerationStructureBuildInput, // vertex, index, transform or instance buffers a build reads
		AccelerationStructureBuildRead, // BLASes a TLAS build references, or the source of an update
		AccelerationStructureBuildWrite, // the structure a build writes
		AccelerationStructureScratch, // a build's scratch buffer
		AccelerationStructureCopySource,
		AccelerationStructureCopyDest,
		AccelerationStructurePostbuildInfoRead, // the structure EmitRaytracingAccelerationStructurePostbuildInfo
												// queries
		AccelerationStructurePostbuildInfoWrite, // the buffer postbuild info is written to, by a build or an emit
	};
	inline constexpr uint32_t kAccessCount = static_cast<uint32_t>(Access::AccelerationStructurePostbuildInfoWrite) + 1;

	const char *ToString(Access access);
	const char *ToString(QueueType queue);
	bool IsWriteAccess(Access access);
	bool IsAccessAllowedOnQueue(Access access, QueueType queue);
	// The barrier state for one access. Buffers have no layout, so theirs is Undefined.
	ResourceState GetAccessState(Access access, bool isTexture);

	// The order here is the order passes are sorted in inside a batch.
	enum class PassKind : uint8_t
	{
		Copy,
		Compute,
		Graphics
	};

	struct ResourceId
	{
		uint32_t index = UINT32_MAX;

		bool IsValid() const
		{
			return index != UINT32_MAX;
		}

		bool operator==(const ResourceId &) const = default;
	};

	struct PassId
	{
		uint32_t index = UINT32_MAX;

		bool IsValid() const
		{
			return index != UINT32_MAX;
		}

		bool operator==(const PassId &) const = default;
	};

	class RenderGraph;

	// What a pass callback gets: the RenderContext to record with (already set to the pass's queue)
	// and the real handles of the resources the pass attached.
	class PassContext
	{
	public:
		PassContext(RenderContext &renderContext, const RenderGraph &graph, PassId pass) :
			rc(renderContext), graph_(graph), pass_(pass)
		{
		}

		RenderContext &rc;

		// Asserts that the pass attached resource: a pass may only touch what it declared.
		// A transient texture or a mip comes as the view the access needs: a UAV if the pass writes it, otherwise an SRV.
		ResourceHandle Get(ResourceId resource) const;
		// For render targets and clears; a mip carries its mip slice.
		Resource GetResource(ResourceId resource) const;
		PassId GetPass() const
		{
			return pass_;
		}
		QueueType GetQueue() const;

	private:
		const RenderGraph &graph_;
		PassId pass_;
	};

	using PassExecuteFn = std::function<void(PassContext &)>;

	struct Attachment
	{
		ResourceId resource;
		Access access = Access::ComputeShaderRead;
		// Back to back concurrent writes share a group: no barrier between them, same batch allowed.
		// Only use it for writes that cannot overlap, such as disjoint ranges.
		bool concurrent = false;
	};

	// Pass declaration, built fluently:
	//   graph.AddPass(rg::Pass::Compute("Cull").OnQueue(QueueType::Compute).Reads(a).Writes(b).Executes(fn));
	struct Pass
	{
		static Pass Compute(std::string name)
		{
			return Pass(std::move(name), PassKind::Compute);
		}

		static Pass Graphics(std::string name)
		{
			return Pass(std::move(name), PassKind::Graphics);
		}

		static Pass Copy(std::string name)
		{
			return Pass(std::move(name), PassKind::Copy);
		}

		Pass &OnQueue(QueueType q)
		{
			queue = q;
			return *this;
		}

		Pass &Use(ResourceId resource, Access access)
		{
			attachments.push_back({ resource, access, false });
			return *this;
		}
		
		Pass &UseConcurrent(ResourceId resource, Access access)
		{
			attachments.push_back({ resource, access, true });
			return *this;
		}
		// Access comes from the pass kind: compute uses SRV/UAV, graphics pixel shader read/colour
		// attachment, copy source/destination. Ray tracing passes are compute passes that Use() their accesses.
		Pass &Reads(ResourceId resource);
		Pass &Writes(ResourceId resource);
		Pass &Executes(PassExecuteFn fn)
		{
			execute = std::move(fn);
			return *this;
		}

		std::string name;
		PassKind kind = PassKind::Compute;
		QueueType queue = QueueType::Graphics;
		std::vector<Attachment> attachments;
		PassExecuteFn execute;

	private:
		Pass(std::string passName, PassKind passKind) : name(std::move(passName)), kind(passKind) {}
	};

	struct CompileOptions
	{
		// Put each pass in the earliest batch its dependencies allow. Otherwise, use linear batching:
		// a pass joins the current batch unless it depends on something in it.
		bool reorderPasses = true;
		// Let transient resources whose lifetimes do not overlap share memory.
		bool aliasTransientMemory = true;
		// After reordering, move passes later (back to front) where that shrinks transient lifetimes
		// without adding batches.
		bool optimizeLifetimes = true;
		// Sort passes inside each batch by kind (copy, compute, raster).
		bool sortBatchesByPassKind = true;
	};

	enum class BarrierKind : uint8_t
	{
		Transition, // between two accesses on the same queue
		Initial, // first use of a transient resource: contents are undefined (discarded)
		QueueRelease, // last use on one queue, before another queue takes the resource
		QueueAcquire, // first use on a queue after another queue released the resource
	};

	struct BarrierInfo
	{
		ResourceId resource;
		BarrierKind kind = BarrierKind::Transition;
		ResourceState before;
		ResourceState after;
		// For imported resources, the before-state is read from the RenderContext when Execute runs.
		bool beforeFromTrackedState = false;
	};

	struct CompiledBatch
	{
		uint32_t section = 0;
		QueueType queue = QueueType::Graphics;
		uint32_t index = 0;
		std::vector<PassId> passes;
		// Recorded right before the batch, in one RecordBarriers call.
		std::vector<BarrierInfo> barriers;
	};

	// Before waitingSection runs on waitingQueue, that queue waits for the submit that ends
	// signalSection on signalQueue.
	struct QueueWaitInfo
	{
		QueueType waitingQueue = QueueType::Graphics;
		uint32_t waitingSection = 0;
		QueueType signalQueue = QueueType::Graphics;
		uint32_t signalSection = 0;
	};

	struct TransientAllocation
	{
		uint64_t offset = 0;
		uint64_t size = 0;
		uint64_t alignment = 0;
		// Inclusive lifetime, in global batch indices (sections laid end to end).
		uint32_t firstBatch = 0;
		uint32_t lastBatch = 0;
	};

	enum class ExecutionEventType : uint8_t
	{
		FrameStartWait, // frame-to-frame wait between queues (multi-queue graphs only)
		Wait, // GPU-side wait on another queue's submit
		Submit,
		Barriers,
		Pass,
	};

	// One step Execute() took, kept so tests and tools can check what was recorded, in order.
	struct ExecutionEvent
	{
		ExecutionEventType type = ExecutionEventType::Pass;
		QueueType queue = QueueType::Graphics;
		uint32_t section = 0;
		PassId pass; // Pass events
		uint32_t barrierCount = 0; // Barriers events
		QueueType otherQueue = QueueType::Graphics; // Wait events: the queue waited on
		uint32_t otherSection = 0; // Wait events: the section waited on
	};

	class RenderGraph
	{
	public:
		// Section 0 is an empty prologue where imported resources come from graphics, each Submit() starts
		// a new section, and an epilogue after the last one returns imported resources to graphics.
		static constexpr uint32_t kPrologueSection = 0;
		static constexpr uint32_t kFirstUserSection = 1;

		RenderGraph() = default;
		~RenderGraph();
		RenderGraph(const RenderGraph &) = delete;
		RenderGraph &operator=(const RenderGraph &) = delete;

		// Transient resources are owned by the graph and placed in its heap; their contents do not
		// survive between executions.
		ResourceId CreateBuffer(const BufferDesc &desc, ResourceViewType viewType = ResourceViewType::UnorderedAccess);
		ResourceId CreateTexture(const TextureDesc &desc, ResourceViewType viewType = ResourceViewType::ShaderResource);
		// Resources owned outside the graph, which keep their contents. Data uploaded once and never written
		// again (meshes, model textures) needs no barriers, so it stays out of the graph.
		ResourceId ImportBuffer(ResourceHandle handle, std::string name = {});
		ResourceId ImportTexture(ResourceHandle handle, std::string name = {});
		// Points an import at another resource of the same kind, such as this frame's back buffer.
		void RebindImport(ResourceId resource, ResourceHandle handle);
		// One mip of a texture, used like any other resource, so each barrier covers only that mip.
		// Once a mip of a texture is used, attaching the whole texture attaches every mip.
		ResourceId Mip(ResourceId texture, uint32_t mip);

		PassId AddPass(Pass pass);
		// Ends the current section: a submit point. Resources can only change queues here.
		void Submit();

		// Validates, schedules, places transient memory and computes barriers and queue syncs. Returns
		// false and sets GetError() if the graph is invalid. Needs the context for allocation sizes.
		bool Compile(RenderContext &rc, const CompileOptions &options = {});
		const std::string &GetError() const
		{
			return error_;
		}
		bool IsCompiled() const
		{
			return compiled_;
		}

		// Call inside a frame with the graphics list open; it leaves graphics open and compute submitted.
		// Creates the transient heap and resources on first use.
		void Execute(RenderContext &rc);
		// Destroys the transient resources and heap. The GPU must be done with them.
		void ReleaseResources();

		// Introspection
		uint32_t GetSectionCount() const
		{
			return static_cast<uint32_t>(sections_.size());
		}
		uint32_t GetEpilogueSection() const
		{
			return GetSectionCount() - 1;
		}
		uint32_t GetBatchCount(uint32_t section, QueueType queue) const;
		uint32_t GetTotalBatchCount() const;
		const CompiledBatch &GetBatch(uint32_t section, QueueType queue, uint32_t index) const;
		std::span<const BarrierInfo> GetQueueReleases(uint32_t section, QueueType queue) const;
		// The queue submits at the end of the section, because another queue waits on it.
		bool IsSubmitSignalled(uint32_t section, QueueType queue) const;
		const std::vector<QueueWaitInfo> &GetQueueWaits() const
		{
			return queueWaits_;
		}

		uint32_t GetPassBatch(PassId pass) const;
		uint32_t GetPassSection(PassId pass) const;
		QueueType GetPassQueue(PassId pass) const;
		const std::string &GetPassName(PassId pass) const;
		bool PassUses(PassId pass, ResourceId resource) const;

		std::optional<TransientAllocation> GetTransientAllocation(ResourceId resource) const;
		uint64_t GetTransientMemorySize() const
		{
			return transientMemorySize_;
		}
		// Sum over transient resources of size x lifetime in batches: what wiggling minimises.
		uint64_t GetVirtualArea() const;

		// The real handle; valid for transient resources once Execute has created them.
		ResourceHandle GetHandle(ResourceId resource) const;

		const std::vector<ExecutionEvent> &GetExecutionTrace() const
		{
			return trace_;
		}
		std::string Dump() const;
		// A self-contained HTML page of the compiled graph: queue swimlane, transient memory and barrier table.
		std::string DumpHtml(const std::string &title) const;

	private:
		static constexpr uint32_t kNoNode = UINT32_MAX;

		struct TextureViews
		{
			ResourceHandle texture = InvalidResourceHandle;
			ResourceHandle srv = InvalidResourceHandle;
			ResourceHandle uav = InvalidResourceHandle;
		};

		struct ResourceNode
		{
			std::string name;
			bool imported = false;
			bool isTexture = false;
			BufferDesc bufferDesc;
			TextureDesc textureDesc;
			ResourceViewType viewType = ResourceViewType::UnorderedAccess;
			ResourceHandle handle = InvalidResourceHandle;
			std::vector<uint32_t> timeline; // access group indices, in recording order

			// Mips: a mip node records accesses, while its texture keeps the memory and the handle.
			uint32_t parent = kNoNode;
			uint32_t mip = 0;
			std::vector<uint32_t> mips; // mip nodes by mip, kNoNode for one not created yet
			mutable std::vector<TextureViews> views; // SRV and UAV views per texture handle, made on first use
			ResourceState trackedState; // imports: the context's state when Execute starts

			// Transient memory.
			uint64_t size = 0;
			uint64_t alignment = 0;
			uint64_t offset = 0;
			uint32_t firstBatch = 0;
			uint32_t lastBatch = 0;
		};

		struct AccessGroup
		{
			uint32_t resource = 0;
			uint32_t timelineIndex = 0;
			uint32_t section = 0;
			QueueType queue = QueueType::Graphics;
			bool isWrite = false;
			bool concurrent = false;
			Access firstAccess = Access::ComputeShaderRead;
			ResourceState state;
			std::vector<uint32_t> passes;
			uint32_t minBatch = 0;
			uint32_t maxBatch = 0;
		};

		struct PassNode
		{
			Pass desc;
			uint32_t section = 0;
			uint32_t batch = 0;
			std::vector<uint32_t> groups; // one access group per attachment, or per mip of a texture used by mip
		};

		struct QueueSection
		{
			std::vector<CompiledBatch> batches;
			std::vector<BarrierInfo> releases; // recorded after the section's last batch on this queue
			std::vector<QueueWaitInfo> waits; // done before the section's first batch on this queue
			bool signal = false;
			bool active = false;
		};

		struct Section
		{
			QueueSection queues[kQueueTypeCount];
			uint32_t batchCount[kQueueTypeCount] = {};
			uint32_t globalStart = 0;
			uint32_t length = 0; // batches: the max over queues
			bool multiQueue = false;
		};

		ResourceId AddResource(ResourceNode node);
		uint32_t AddMipNode(uint32_t texture, uint32_t mip);
		// Creates every mip node of each texture used by mip, once mip counts are known.
		bool ResolveMips(RenderContext &rc);
		// The node that owns the memory and handle: the texture of a mip, otherwise the node itself.
		uint32_t Owner(uint32_t resource) const
		{
			return resources_[resource].parent == kNoNode ? resource : resources_[resource].parent;
		}
		// Accessed by some pass, directly or through its mips.
		bool IsUsed(const ResourceNode &node) const;
		// Placed in transient memory: a used transient that is not a mip.
		bool OwnsTransientMemory(const ResourceNode &node) const
		{
			return !node.imported && node.parent == kNoNode && IsUsed(node);
		}
		// SRV or UAV view of a mip or of a whole texture, made on first use.
		ResourceHandle GetView(RenderContext &rc, uint32_t resource, bool unorderedAccess) const;
		// The access pass uses resource with, directly or through the whole texture.
		Access GetPassAccess(PassId pass, ResourceId resource) const;
		void ResetCompiledState();
		bool Fail(std::string message);
		bool BuildTimelines();
		// Records one attachment on one node (a texture used by mip records it on each mip).
		bool AddAccess(uint32_t pass, uint32_t resource, const Attachment &attachment);
		void SchedulePasses(const CompileOptions &options);
		void ComputeSectionLayout();
		void RecomputeGroupBatches(uint32_t group);
		// Inclusive [first, last] global batch range of a resource's accesses.
		std::pair<uint32_t, uint32_t> ComputeLifetime(const ResourceNode &node) const;
		uint64_t ResourceArea(const ResourceNode &node) const;
		void OptimizeLifetimes();
		void BuildBatches(const CompileOptions &options);
		void PlaceTransientMemory(bool alias);
		void BuildBarriersAndSyncs();
		void AddQueueWait(QueueType waitingQueue,
						  uint32_t waitingSection,
						  QueueType signalQueue,
						  uint32_t signalSection);
		void FinaliseQueueWaits();
		void CreateTransientResources(RenderContext &rc);
		ResourceTransitionDesc ResolveBarrier(const BarrierInfo &barrier) const;
		void RecordBarrierList(RenderContext &rc,
							   std::span<const BarrierInfo> barriers,
							   QueueType queue,
							   uint32_t section);

		static uint32_t QueueIndex(QueueType queue)
		{
			return static_cast<uint32_t>(queue);
		}

		std::vector<ResourceNode> resources_;
		std::vector<PassNode> passes_;
		uint32_t currentSection_ = kFirstUserSection;

		// Compiled state.
		bool compiled_ = false;
		std::string error_;
		std::vector<AccessGroup> groups_;
		std::vector<Section> sections_;
		std::vector<QueueWaitInfo> pendingWaits_;
		std::vector<QueueWaitInfo> queueWaits_;
		uint64_t transientMemorySize_ = 0;
		bool usesComputeQueue_ = false;

		// Realised state.
		RenderContext *realisedContext_ = nullptr;
		MemoryHeapHandle heap_ = InvalidMemoryHeapHandle;
		std::vector<ExecutionEvent> trace_;

		friend class PassContext;
	};
} // namespace rg
