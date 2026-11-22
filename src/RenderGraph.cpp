#include "RenderGraph.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace rg
{
	namespace
	{
		uint64_t AlignUp(uint64_t value, uint64_t alignment)
		{
			return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
		}

		// The state a resource is left in when one queue hands it to another. Textures go to COMMON,
		// a layout every queue type supports. The fence between the queues provides the dependency.
		ResourceState ReleasedState(bool isTexture)
		{
			return ResourceState{ isTexture ? ResourceLayout::Common : ResourceLayout::Undefined,
								  ResourceSync::None,
								  ResourceAccess::NoAccess };
		}

		enum AccessFlags : uint8_t
		{
			kWrite = 1 << 0,
			kTextureOnly = 1 << 1,
			kBufferOnly = 1 << 2,
			kComputeQueue = 1 << 3, // also allowed on the compute queue
			kUnorderedAccess = 1 << 4, // transient resources need ALLOW_UNORDERED_ACCESS
			kAccelerationStructure = 1 << 5, // the buffer must be an acceleration structure, so imported
		};

		struct AccessInfo
		{
			Access access;
			const char *name;
			ResourceState state; // the layout only applies to textures
			uint8_t flags;
		};

		constexpr ResourceSync kGraphicsShading = ResourceSync::VertexShading | ResourceSync::PixelShading;
		constexpr ResourceSync kBuild = ResourceSync::BuildRaytracingAccelerationStructure;
		constexpr ResourceSync kCopyAS = ResourceSync::CopyRaytracingAccelerationStructure;
		constexpr ResourceSync kEmitAS = ResourceSync::EmitRaytracingAccelerationStructurePostbuildInfo;
		constexpr ResourceAccess kASRead = ResourceAccess::RaytracingAccelerationStructureRead;
		constexpr ResourceAccess kASWrite = ResourceAccess::RaytracingAccelerationStructureWrite;
		constexpr ResourceLayout kSRV = ResourceLayout::ShaderResource;
		constexpr ResourceLayout kUAV = ResourceLayout::UnorderedAccess;
		constexpr ResourceLayout kNone = ResourceLayout::Undefined;

		// In Access order; Info() checks it.
		const AccessInfo kAccessInfo[] = {
			{ Access::ComputeShaderRead,
			  "ComputeShaderRead",
			  { kSRV, ResourceSync::ComputeShading, ResourceAccess::ShaderResource },
			  kComputeQueue },
			{ Access::ComputeShaderWrite,
			  "ComputeShaderWrite",
			  { kUAV, ResourceSync::ComputeShading, ResourceAccess::UnorderedAccess },
			  kWrite | kComputeQueue | kUnorderedAccess },
			{ Access::ComputeShaderReadWrite,
			  "ComputeShaderReadWrite",
			  { kUAV, ResourceSync::ComputeShading, ResourceAccess::UnorderedAccess },
			  kWrite | kComputeQueue | kUnorderedAccess },
			{ Access::ClearUnorderedAccess,
			  "ClearUnorderedAccess",
			  { kUAV, ResourceSync::ClearUnorderedAccessView, ResourceAccess::UnorderedAccess },
			  kWrite | kComputeQueue | kUnorderedAccess },

			{ Access::VertexShaderRead,
			  "VertexShaderRead",
			  { kSRV, ResourceSync::VertexShading, ResourceAccess::ShaderResource },
			  0 },
			{ Access::VertexShaderWrite,
			  "VertexShaderWrite",
			  { kUAV, ResourceSync::VertexShading, ResourceAccess::UnorderedAccess },
			  kWrite | kUnorderedAccess },
			{ Access::MeshShaderRead,
			  "MeshShaderRead",
			  { kSRV, ResourceSync::VertexShading, ResourceAccess::ShaderResource },
			  0 },
			{ Access::MeshShaderWrite,
			  "MeshShaderWrite",
			  { kUAV, ResourceSync::VertexShading, ResourceAccess::UnorderedAccess },
			  kWrite | kUnorderedAccess },
			{ Access::PixelShaderRead,
			  "PixelShaderRead",
			  { kSRV, ResourceSync::PixelShading, ResourceAccess::ShaderResource },
			  0 },
			{ Access::PixelShaderWrite,
			  "PixelShaderWrite",
			  { kUAV, ResourceSync::PixelShading, ResourceAccess::UnorderedAccess },
			  kWrite | kUnorderedAccess },
			{ Access::GraphicsShaderRead,
			  "GraphicsShaderRead",
			  { kSRV, kGraphicsShading, ResourceAccess::ShaderResource },
			  0 },
			{ Access::GraphicsShaderWrite,
			  "GraphicsShaderWrite",
			  { kUAV, kGraphicsShading, ResourceAccess::UnorderedAccess },
			  kWrite | kUnorderedAccess },
			{ Access::ConstantBufferRead,
			  "ConstantBufferRead",
			  { kNone, ResourceSync::AllShading, ResourceAccess::ConstantBuffer },
			  kBufferOnly | kComputeQueue },

			{ Access::IndirectArgumentRead,
			  "IndirectArgumentRead",
			  { kNone, ResourceSync::ExecuteIndirect, ResourceAccess::IndirectArgument },
			  kBufferOnly | kComputeQueue },
			{ Access::PredicationRead,
			  "PredicationRead",
			  { kNone, ResourceSync::Predication, ResourceAccess::Predication },
			  kBufferOnly | kComputeQueue },
			{ Access::ShadingRateSource,
			  "ShadingRateSource",
			  { ResourceLayout::ShadingRateSource, ResourceSync::PixelShading, ResourceAccess::ShadingRateSource },
			  kTextureOnly },
			{ Access::CopySource,
			  "CopySource",
			  { ResourceLayout::CopySource, ResourceSync::Copy, ResourceAccess::CopySource },
			  kComputeQueue },
			{ Access::CopyDest,
			  "CopyDest",
			  { ResourceLayout::CopyDest, ResourceSync::Copy, ResourceAccess::CopyDest },
			  kWrite | kComputeQueue },
			{ Access::ColourAttachment,
			  "ColourAttachment",
			  { ResourceLayout::RenderTarget, ResourceSync::RenderTarget, ResourceAccess::RenderTarget },
			  kWrite | kTextureOnly },
			{ Access::DepthStencilWrite,
			  "DepthStencilWrite",
			  { ResourceLayout::DepthStencilWrite, ResourceSync::DepthStencil, ResourceAccess::DepthStencilWrite },
			  kWrite | kTextureOnly },
			{ Access::DepthStencilRead,
			  "DepthStencilRead",
			  { ResourceLayout::DepthStencilRead, ResourceSync::DepthStencil, ResourceAccess::DepthStencilRead },
			  kTextureOnly },

			{ Access::RaytracingShaderRead,
			  "RaytracingShaderRead",
			  { kSRV, ResourceSync::Raytracing, ResourceAccess::ShaderResource },
			  kComputeQueue },
			{ Access::RaytracingShaderWrite,
			  "RaytracingShaderWrite",
			  { kUAV, ResourceSync::Raytracing, ResourceAccess::UnorderedAccess },
			  kWrite | kComputeQueue | kUnorderedAccess },
			{ Access::AccelerationStructureRead,
			  "AccelerationStructureRead",
			  { kNone, ResourceSync::Raytracing | ResourceSync::AllShading, kASRead },
			  kBufferOnly | kComputeQueue | kAccelerationStructure },
			{ Access::AccelerationStructureBuildInput,
			  "AccelerationStructureBuildInput",
			  { kNone, kBuild, ResourceAccess::ShaderResource },
			  kBufferOnly | kComputeQueue },
			{ Access::AccelerationStructureBuildRead,
			  "AccelerationStructureBuildRead",
			  { kNone, kBuild, kASRead },
			  kBufferOnly | kComputeQueue | kAccelerationStructure },
			{ Access::AccelerationStructureBuildWrite,
			  "AccelerationStructureBuildWrite",
			  { kNone, kBuild, kASWrite },
			  kWrite | kBufferOnly | kComputeQueue | kAccelerationStructure },
			{ Access::AccelerationStructureScratch,
			  "AccelerationStructureScratch",
			  { kNone, kBuild, ResourceAccess::UnorderedAccess },
			  kWrite | kBufferOnly | kComputeQueue | kUnorderedAccess },
			{ Access::AccelerationStructureCopySource,
			  "AccelerationStructureCopySource",
			  { kNone, kCopyAS, kASRead },
			  kBufferOnly | kComputeQueue | kAccelerationStructure },
			{ Access::AccelerationStructureCopyDest,
			  "AccelerationStructureCopyDest",
			  { kNone, kCopyAS, kASWrite },
			  kWrite | kBufferOnly | kComputeQueue | kAccelerationStructure },
			{ Access::AccelerationStructurePostbuildInfoRead,
			  "AccelerationStructurePostbuildInfoRead",
			  { kNone, kEmitAS, kASRead },
			  kBufferOnly | kComputeQueue | kAccelerationStructure },
			{ Access::AccelerationStructurePostbuildInfoWrite,
			  "AccelerationStructurePostbuildInfoWrite",
			  { kNone, kBuild | kEmitAS, ResourceAccess::UnorderedAccess },
			  kWrite | kBufferOnly | kComputeQueue | kUnorderedAccess },
		};

		const AccessInfo &Info(Access access)
		{
			const size_t index = static_cast<size_t>(access);
			assert(index < std::size(kAccessInfo) && kAccessInfo[index].access == access &&
				   "kAccessInfo is out of order");
			return kAccessInfo[index];
		}

		bool HasFlag(Access access, AccessFlags flag)
		{
			return (Info(access).flags & flag) != 0;
		}

		// Reads share a group only in one layout or the shader and copy read layouts, which GENERIC_READ covers.
		// Depth test reads and shading rate images have layouts of their own.
		bool CanMergeReadLayouts(ResourceLayout a, ResourceLayout b)
		{
			const auto generic = [](ResourceLayout l)
			{
				return l == ResourceLayout::ShaderResource || l == ResourceLayout::CopySource ||
					   l == ResourceLayout::GenericRead;
			};
			return a == b || (generic(a) && generic(b));
		}

		// One layout that every read in a merged read group can use.
		ResourceLayout MergeReadLayouts(ResourceLayout a, ResourceLayout b)
		{
			return a == b ? a : ResourceLayout::GenericRead;
		}

		bool SameState(const ResourceState &a, const ResourceState &b)
		{
			return a.layout == b.layout && a.sync == b.sync && a.access == b.access;
		}

		uint32_t FullMipChain(const TextureDesc &desc)
		{
			uint32_t size = std::max(desc.width, desc.height);
			if (desc.dimension == TextureDimension::Texture3D)
			{
				size = std::max(size, desc.depthOrArraySize);
			}
			uint32_t levels = 1;
			while (size > 1)
			{
				size /= 2;
				++levels;
			}
			return levels;
		}

		bool RangesOverlap(uint64_t aBegin, uint64_t aSize, uint64_t bBegin, uint64_t bSize)
		{
			return aBegin < bBegin + bSize && bBegin < aBegin + aSize;
		}

		const char *ToString(BarrierKind kind)
		{
			switch (kind)
			{
				case BarrierKind::Transition:
					return "transition";
				case BarrierKind::Initial:
					return "initial";
				case BarrierKind::QueueRelease:
					return "release";
				case BarrierKind::QueueAcquire:
					return "acquire";
			}
			return "?";
		}
	} // namespace

	const char *ToString(Access access)
	{
		return Info(access).name;
	}

	const char *ToString(QueueType queue)
	{
		return queue == QueueType::Compute ? "Compute" : "Graphics";
	}

	bool IsWriteAccess(Access access)
	{
		return HasFlag(access, kWrite);
	}

	bool IsAccessAllowedOnQueue(Access access, QueueType queue)
	{
		// Graphics stages, attachments and the input assembler do not exist on a compute queue.
		return queue == QueueType::Graphics || HasFlag(access, kComputeQueue);
	}

	ResourceState GetAccessState(Access access, bool isTexture)
	{
		ResourceState state = Info(access).state;
		if (!isTexture)
		{
			state.layout = ResourceLayout::Undefined;
		}
		return state;
	}

	// Pass

	Pass &Pass::Reads(ResourceId resource)
	{
		switch (kind)
		{
			case PassKind::Copy:
				return Use(resource, Access::CopySource);
			case PassKind::Graphics:
				return Use(resource, Access::PixelShaderRead);
			default:
				return Use(resource, Access::ComputeShaderRead);
		}
	}

	Pass &Pass::Writes(ResourceId resource)
	{
		switch (kind)
		{
			case PassKind::Copy:
				return Use(resource, Access::CopyDest);
			case PassKind::Graphics:
				return Use(resource, Access::ColourAttachment);
			default:
				return Use(resource, Access::ComputeShaderWrite);
		}
	}

	// PassContext

	ResourceHandle PassContext::Get(ResourceId resource) const
	{
		assert(graph_.PassUses(pass_, resource) && "a pass may only use resources it attached");
		const auto &node = graph_.resources_[resource.index];
		if (node.isTexture)
		{
			// A mip always gets its own view. A transient texture gets one when the access needs the other
			// kind of view than it was created with; an import's view is not known, so its handle is used.
			const bool uav = HasFlag(graph_.GetPassAccess(pass_, resource), kUnorderedAccess);
			const bool ownViewFits = uav == (node.viewType == ResourceViewType::UnorderedAccess);
			if (node.parent != RenderGraph::kNoNode || (!node.imported && !ownViewFits))
			{
				return graph_.GetView(rc, resource.index, uav);
			}
		}
		const ResourceHandle handle = graph_.GetHandle(resource);
		assert(handle != InvalidResourceHandle);
		return handle;
	}

	Resource PassContext::GetResource(ResourceId resource) const
	{
		assert(graph_.PassUses(pass_, resource) && "a pass may only use resources it attached");
		const ResourceHandle handle = graph_.GetHandle(resource);
		assert(handle != InvalidResourceHandle);
		Resource result = rc.GetResource(handle);
		const auto &node = graph_.resources_[resource.index];
		result.mip = node.parent != RenderGraph::kNoNode ? node.mip : 0;
		return result;
	}

	QueueType PassContext::GetQueue() const
	{
		return graph_.GetPassQueue(pass_);
	}

	// Declaration

	RenderGraph::~RenderGraph()
	{
		ReleaseResources();
	}

	ResourceId RenderGraph::AddResource(ResourceNode node)
	{
		compiled_ = false;
		resources_.push_back(std::move(node));
		return ResourceId{ static_cast<uint32_t>(resources_.size() - 1) };
	}

	ResourceId RenderGraph::CreateBuffer(const BufferDesc &desc, ResourceViewType viewType)
	{
		ResourceNode node;
		node.name = desc.debugName.empty() ? "Buffer" + std::to_string(resources_.size()) : desc.debugName;
		node.isTexture = false;
		node.bufferDesc = desc;
		node.viewType = viewType;
		return AddResource(std::move(node));
	}

	ResourceId RenderGraph::CreateTexture(const TextureDesc &desc, ResourceViewType viewType)
	{
		ResourceNode node;
		node.name = desc.debugName.empty() ? "Texture" + std::to_string(resources_.size()) : desc.debugName;
		node.isTexture = true;
		node.textureDesc = desc;
		node.viewType = viewType;
		return AddResource(std::move(node));
	}

	ResourceId RenderGraph::ImportBuffer(ResourceHandle handle, std::string name)
	{
		assert(handle != InvalidResourceHandle);
		ResourceNode node;
		node.name = name.empty() ? "ImportedBuffer" + std::to_string(resources_.size()) : std::move(name);
		node.imported = true;
		node.isTexture = false;
		node.handle = handle;
		return AddResource(std::move(node));
	}

	ResourceId RenderGraph::ImportTexture(ResourceHandle handle, std::string name)
	{
		assert(handle != InvalidResourceHandle);
		ResourceNode node;
		node.name = name.empty() ? "ImportedTexture" + std::to_string(resources_.size()) : std::move(name);
		node.imported = true;
		node.isTexture = true;
		node.handle = handle;
		return AddResource(std::move(node));
	}

	void RenderGraph::RebindImport(ResourceId resource, ResourceHandle handle)
	{
		assert(resource.index < resources_.size() && resources_[resource.index].imported);
		assert(resources_[resource.index].parent == kNoNode && "rebind the texture, not one of its mips");
		assert(handle != InvalidResourceHandle);
		resources_[resource.index].handle = handle;
	}

	ResourceId RenderGraph::Mip(ResourceId texture, uint32_t mip)
	{
		assert(texture.index < resources_.size());
		const ResourceNode &node = resources_[texture.index];
		assert(node.isTexture && "only textures have mips");
		assert(node.parent == kNoNode && "Mip() takes the whole texture");
		if (mip < node.mips.size() && node.mips[mip] != kNoNode)
		{
			return ResourceId{ node.mips[mip] };
		}
		return ResourceId{ AddMipNode(texture.index, mip) };
	}

	uint32_t RenderGraph::AddMipNode(uint32_t texture, uint32_t mip)
	{
		ResourceNode node;
		node.name = resources_[texture].name + "[mip " + std::to_string(mip) + "]";
		node.imported = resources_[texture].imported;
		node.isTexture = true;
		node.parent = texture;
		node.mip = mip;
		const uint32_t index = AddResource(std::move(node)).index;

		std::vector<uint32_t> &mips = resources_[texture].mips;
		if (mips.size() <= mip)
		{
			mips.resize(mip + 1, kNoNode);
		}
		mips[mip] = index;
		return index;
	}

	bool RenderGraph::ResolveMips(RenderContext &rc)
	{
		const uint32_t count = static_cast<uint32_t>(resources_.size());
		for (uint32_t r = 0; r < count; ++r)
		{
			if (resources_[r].mips.empty())
			{
				continue;
			}
			const TextureDesc desc =
					resources_[r].imported ? rc.GetTextureDesc(resources_[r].handle) : resources_[r].textureDesc;
			const uint32_t levels = desc.mipLevels != 0 ? desc.mipLevels : FullMipChain(desc);
			const uint32_t requested = static_cast<uint32_t>(resources_[r].mips.size());
			if (requested > levels)
			{
				return Fail("'" + resources_[r].name + "' has " + std::to_string(levels) + " mips, so it has no mip " +
							std::to_string(requested - 1));
			}
			for (uint32_t m = 0; m < levels; ++m)
			{
				if (m >= resources_[r].mips.size() || resources_[r].mips[m] == kNoNode)
				{
					AddMipNode(r, m);
				}
			}
		}
		return true;
	}

	bool RenderGraph::IsUsed(const ResourceNode &node) const
	{
		if (!node.timeline.empty())
		{
			return true;
		}
		for (uint32_t m: node.mips)
		{
			if (m != kNoNode && !resources_[m].timeline.empty())
			{
				return true;
			}
		}
		return false;
	}

	PassId RenderGraph::AddPass(Pass pass)
	{
		compiled_ = false;
		PassNode node{ std::move(pass), currentSection_ };
		passes_.push_back(std::move(node));
		return PassId{ static_cast<uint32_t>(passes_.size() - 1) };
	}

	void RenderGraph::Submit()
	{
		compiled_ = false;
		++currentSection_;
	}

	// Compile

	void RenderGraph::ResetCompiledState()
	{
		compiled_ = false;
		error_.clear();
		groups_.clear();
		sections_.clear();
		pendingWaits_.clear();
		queueWaits_.clear();
		transientMemorySize_ = 0;
		usesComputeQueue_ = false;
		for (PassNode &pass: passes_)
		{
			pass.batch = 0;
			pass.groups.clear();
		}
		for (ResourceNode &node: resources_)
		{
			node.timeline.clear();
			node.offset = 0;
			node.firstBatch = 0;
			node.lastBatch = 0;
		}
	}

	bool RenderGraph::Fail(std::string message)
	{
		error_ = std::move(message);
		compiled_ = false;
		return false;
	}

	bool RenderGraph::Compile(RenderContext &rc, const CompileOptions &options)
	{
		if (realisedContext_)
		{
			return Fail("Compile: call ReleaseResources() (with the GPU idle) before recompiling an executed graph");
		}

		ResetCompiledState();
		// Prologue, user sections, epilogue.
		sections_.resize(currentSection_ + 2);

		if (!ResolveMips(rc) || !BuildTimelines())
		{
			return false;
		}

		for (ResourceNode &node: resources_)
		{
			if (!OwnsTransientMemory(node))
			{
				continue;
			}
			const ResourceAllocationInfo info = node.isTexture ? rc.GetTextureAllocationInfo(node.textureDesc)
															   : rc.GetBufferAllocationInfo(node.bufferDesc);
			node.size = info.sizeInBytes;
			node.alignment = info.alignment;
		}

		SchedulePasses(options);
		ComputeSectionLayout();
		if (options.reorderPasses && options.optimizeLifetimes)
		{
			OptimizeLifetimes();
		}
		BuildBatches(options);
		PlaceTransientMemory(options.aliasTransientMemory);
		BuildBarriersAndSyncs();
		FinaliseQueueWaits();

		compiled_ = true;
		return true;
	}

	bool RenderGraph::BuildTimelines()
	{
		// Section and queue each memory owner was last used on, to keep one owner off two queues at once.
		std::vector<std::pair<uint32_t, QueueType>> lastUse(resources_.size(), { UINT32_MAX, QueueType::Graphics });

		for (uint32_t passIndex = 0; passIndex < passes_.size(); ++passIndex)
		{
			PassNode &pass = passes_[passIndex];
			const Pass &desc = pass.desc;

			if (desc.kind == PassKind::Graphics && desc.queue == QueueType::Compute)
			{
				return Fail("pass '" + desc.name + "' is a raster pass on the compute queue");
			}

			for (uint32_t a = 0; a < desc.attachments.size(); ++a)
			{
				const Attachment &attachment = desc.attachments[a];
				if (!attachment.resource.IsValid() || attachment.resource.index >= resources_.size())
				{
					return Fail("pass '" + desc.name + "' attaches an invalid resource");
				}
				const uint32_t r = attachment.resource.index;
				for (uint32_t b = 0; b < a; ++b)
				{
					const uint32_t other = desc.attachments[b].resource.index;
					if (other == r)
					{
						return Fail("pass '" + desc.name + "' attaches '" + resources_[r].name +
									"' more than once; use a single read-write access instead");
					}
					if (resources_[r].parent == other || resources_[other].parent == r)
					{
						return Fail("pass '" + desc.name + "' attaches '" + resources_[other].name + "' and '" +
									resources_[r].name + "'; attach a texture or its mips, not both");
					}
				}

				const ResourceNode &node = resources_[r];
				const Access access = attachment.access;
				if (HasFlag(access, kTextureOnly) && !node.isTexture)
				{
					return Fail("pass '" + desc.name + "' uses buffer '" + node.name + "' as " + ToString(access));
				}
				if (HasFlag(access, kBufferOnly) && node.isTexture)
				{
					return Fail("pass '" + desc.name + "' uses texture '" + node.name + "' as " + ToString(access));
				}
				if (HasFlag(access, kAccelerationStructure) && !node.imported)
				{
					return Fail("pass '" + desc.name + "' uses transient '" + node.name + "' as " + ToString(access) +
								"; acceleration structures come from the RenderContext, so import them");
				}
				if (!IsAccessAllowedOnQueue(access, desc.queue))
				{
					return Fail("pass '" + desc.name + "' uses " + ToString(access) + " on the " +
								ToString(desc.queue) + " queue");
				}

				const uint32_t owner = Owner(r);
				if (lastUse[owner].first == pass.section && lastUse[owner].second != desc.queue)
				{
					return Fail("resource '" + resources_[owner].name +
								"' is used on both the graphics and compute queues in section " +
								std::to_string(pass.section) + " (pass '" + desc.name +
								"'); put a Submit() between them");
				}
				lastUse[owner] = { pass.section, desc.queue };

				// The graph knows how transient resources are used, so it sets their usage flags.
				ResourceNode &ownerNode = resources_[owner];
				if (!ownerNode.imported)
				{
					if (ownerNode.isTexture)
					{
						ResourceUsageFlags &usage = ownerNode.textureDesc.usageFlags;
						if (HasFlag(access, kUnorderedAccess) ||
							ownerNode.viewType == ResourceViewType::UnorderedAccess)
							usage = usage | ResourceUsageFlags::UnorderedAccess;
						if (access == Access::ColourAttachment)
							usage = usage | ResourceUsageFlags::RenderTarget;
						if (access == Access::DepthStencilWrite || access == Access::DepthStencilRead)
							usage = usage | ResourceUsageFlags::DepthStencil;
					}
					else if (HasFlag(access, kUnorderedAccess) ||
							 ownerNode.viewType == ResourceViewType::UnorderedAccess)
					{
						ownerNode.bufferDesc.usageFlags =
								ownerNode.bufferDesc.usageFlags | ResourceUsageFlags::UnorderedAccess;
					}
				}

				// A texture used by mip records every access on its mips.
				std::vector<uint32_t> targets = resources_[r].mips;
				if (targets.empty())
				{
					targets.push_back(r);
				}
				for (uint32_t t: targets)
				{
					if (!AddAccess(passIndex, t, attachment))
					{
						return false;
					}
				}
			}
		}
		return true;
	}

	bool RenderGraph::AddAccess(uint32_t passIndex, uint32_t resource, const Attachment &attachment)
	{
		PassNode &pass = passes_[passIndex];
		ResourceNode &node = resources_[resource];
		const Access access = attachment.access;
		const bool isWrite = IsWriteAccess(access);
		if (node.timeline.empty() && !node.imported && !isWrite)
		{
			return Fail("pass '" + pass.desc.name + "' reads transient resource '" + node.name +
						"' before anything writes it");
		}

		const ResourceState state = GetAccessState(access, node.isTexture);
		if (!node.timeline.empty())
		{
			AccessGroup &last = groups_[node.timeline.back()];
			const bool sameQueueSection = last.section == pass.section && last.queue == pass.desc.queue;
			const bool mergeRead = !isWrite && !last.isWrite &&
								   (!node.isTexture || CanMergeReadLayouts(last.state.layout, state.layout));
			const bool mergeWrite =
					isWrite && last.isWrite && attachment.concurrent && last.concurrent && last.firstAccess == access;
			if (sameQueueSection && (mergeRead || mergeWrite))
			{
				last.passes.push_back(passIndex);
				if (mergeRead)
				{
					last.state.sync = last.state.sync | state.sync;
					last.state.access = last.state.access | state.access;
					last.state.layout = node.isTexture ? MergeReadLayouts(last.state.layout, state.layout)
													   : ResourceLayout::Undefined;
				}
				pass.groups.push_back(node.timeline.back());
				return true;
			}
		}

		AccessGroup group;
		group.resource = resource;
		group.timelineIndex = static_cast<uint32_t>(node.timeline.size());
		group.section = pass.section;
		group.queue = pass.desc.queue;
		group.isWrite = isWrite;
		group.concurrent = isWrite && attachment.concurrent;
		group.firstAccess = access;
		group.state = state;
		group.passes.push_back(passIndex);
		groups_.push_back(std::move(group));

		const uint32_t groupIndex = static_cast<uint32_t>(groups_.size() - 1);
		node.timeline.push_back(groupIndex);
		pass.groups.push_back(groupIndex);
		return true;
	}

	void RenderGraph::SchedulePasses(const CompileOptions &options)
	{
		// Linear batching: the batch most recently opened on each (section, queue).
		std::vector<std::array<uint32_t, kQueueTypeCount>> currentBatch(sections_.size(), { 0, 0 });

		for (uint32_t passIndex = 0; passIndex < passes_.size(); ++passIndex)
		{
			PassNode &pass = passes_[passIndex];
			const uint32_t q = QueueIndex(pass.desc.queue);

			// One past the latest batch of the group before each of the pass's groups.
			// Earlier sections are already ordered by the submit.
			uint32_t needed = 0;
			for (uint32_t g: pass.groups)
			{
				const AccessGroup &group = groups_[g];
				if (group.timelineIndex == 0)
				{
					continue;
				}
				const AccessGroup &previous = groups_[resources_[group.resource].timeline[group.timelineIndex - 1]];
				if (previous.section == pass.section && previous.queue == pass.desc.queue)
				{
					needed = std::max(needed, previous.maxBatch + 1);
				}
			}

			uint32_t batch = needed;
			if (!options.reorderPasses)
			{
				batch = std::max(needed, currentBatch[pass.section][q]);
				currentBatch[pass.section][q] = batch;
			}
			pass.batch = batch;

			for (uint32_t g: pass.groups)
			{
				AccessGroup &group = groups_[g];
				if (group.passes.front() == passIndex)
				{
					group.minBatch = group.maxBatch = batch;
				}
				else
				{
					group.minBatch = std::min(group.minBatch, batch);
					group.maxBatch = std::max(group.maxBatch, batch);
				}
			}
		}
	}

	void RenderGraph::ComputeSectionLayout()
	{
		for (Section &section: sections_)
		{
			section.batchCount[0] = section.batchCount[1] = 0;
		}
		for (const PassNode &pass: passes_)
		{
			uint32_t &count = sections_[pass.section].batchCount[QueueIndex(pass.desc.queue)];
			count = std::max(count, pass.batch + 1);
		}

		uint32_t start = 0;
		for (Section &section: sections_)
		{
			const uint32_t activeQueues = (section.batchCount[0] > 0 ? 1u : 0u) + (section.batchCount[1] > 0 ? 1u : 0u);
			section.multiQueue = activeQueues > 1;
			section.length = std::max(section.batchCount[0], section.batchCount[1]);
			section.globalStart = start;
			start += section.length;
		}
	}

	void RenderGraph::RecomputeGroupBatches(uint32_t g)
	{
		AccessGroup &group = groups_[g];
		group.minBatch = UINT32_MAX;
		group.maxBatch = 0;
		for (uint32_t passIndex: group.passes)
		{
			group.minBatch = std::min(group.minBatch, passes_[passIndex].batch);
			group.maxBatch = std::max(group.maxBatch, passes_[passIndex].batch);
		}
	}

	std::pair<uint32_t, uint32_t> RenderGraph::ComputeLifetime(const ResourceNode &node) const
	{
		uint32_t first = UINT32_MAX;
		uint32_t last = 0;
		auto add = [&](const ResourceNode &accessed)
		{
			for (uint32_t g: accessed.timeline)
			{
				const AccessGroup &group = groups_[g];
				const Section &section = sections_[group.section];
				uint32_t begin = section.globalStart + group.minBatch;
				uint32_t end = section.globalStart + group.maxBatch;
				if (section.multiQueue)
				{
					// Queues in one section are not ordered against each other, so a resource used there may
					// be live at any point of the section.
					begin = section.globalStart;
					end = section.globalStart + section.length - 1;
				}
				first = std::min(first, begin);
				last = std::max(last, end);
			}
		};
		// A texture used by mip lives from the first access of any mip to the last.
		add(node);
		for (uint32_t m: node.mips)
		{
			if (m != kNoNode)
			{
				add(resources_[m]);
			}
		}
		return { first, last };
	}

	uint64_t RenderGraph::ResourceArea(const ResourceNode &node) const
	{
		if (!OwnsTransientMemory(node))
		{
			return 0;
		}
		const auto [first, last] = ComputeLifetime(node);
		return node.size * (static_cast<uint64_t>(last - first) + 1);
	}

	void RenderGraph::OptimizeLifetimes()
	{
		auto touchedArea = [this](const PassNode &pass)
		{
			// Each memory owner once, though a pass may use several mips of one texture.
			std::vector<uint32_t> owners;
			uint64_t area = 0;
			for (uint32_t g: pass.groups)
			{
				const uint32_t owner = Owner(groups_[g].resource);
				if (std::find(owners.begin(), owners.end(), owner) == owners.end())
				{
					owners.push_back(owner);
					area += ResourceArea(resources_[owner]);
				}
			}
			return area;
		};

		// One back-to-front sweep: a later pass that moves can make room for an earlier one.
		for (uint32_t i = static_cast<uint32_t>(passes_.size()); i-- > 0;)
		{
			PassNode &pass = passes_[i];
			const Section &section = sections_[pass.section];
			if (section.multiQueue)
			{
				continue; // lifetimes there span the whole section anyway
			}
			const uint32_t batchCount = section.batchCount[QueueIndex(pass.desc.queue)];

			while (pass.batch + 1 < batchCount) // never lengthen the critical path
			{
				const uint32_t candidate = pass.batch + 1;

				// Every pass depending on this one (the next group of each of its resources, when that group
				// is in this section) must still come strictly later.
				bool legal = true;
				for (uint32_t g: pass.groups)
				{
					const AccessGroup &group = groups_[g];
					const ResourceNode &node = resources_[group.resource];
					if (group.timelineIndex + 1 >= node.timeline.size())
					{
						continue;
					}
					const AccessGroup &next = groups_[node.timeline[group.timelineIndex + 1]];
					if (next.section != pass.section)
					{
						continue;
					}
					for (uint32_t dependent: next.passes)
					{
						legal = legal && passes_[dependent].batch > candidate;
					}
				}
				if (!legal)
				{
					break;
				}

				const uint64_t before = touchedArea(pass);
				const uint32_t original = pass.batch;
				pass.batch = candidate;
				for (uint32_t g: pass.groups)
				{
					RecomputeGroupBatches(g);
				}
				if (touchedArea(pass) >= before)
				{
					pass.batch = original;
					for (uint32_t g: pass.groups)
					{
						RecomputeGroupBatches(g);
					}
					break;
				}
			}
		}
	}

	void RenderGraph::BuildBatches(const CompileOptions &options)
	{
		for (uint32_t s = 0; s < sections_.size(); ++s)
		{
			for (uint32_t q = 0; q < kQueueTypeCount; ++q)
			{
				QueueSection &queueSection = sections_[s].queues[q];
				queueSection.batches.resize(sections_[s].batchCount[q]);
				for (uint32_t b = 0; b < queueSection.batches.size(); ++b)
				{
					queueSection.batches[b].section = s;
					queueSection.batches[b].queue = static_cast<QueueType>(q);
					queueSection.batches[b].index = b;
				}
			}
		}

		for (uint32_t passIndex = 0; passIndex < passes_.size(); ++passIndex)
		{
			const PassNode &pass = passes_[passIndex];
			sections_[pass.section].queues[QueueIndex(pass.desc.queue)].batches[pass.batch].passes.push_back(
					PassId{ passIndex });
			if (pass.desc.queue == QueueType::Compute)
			{
				usesComputeQueue_ = true;
			}
		}

		if (options.sortBatchesByPassKind)
		{
			// Any order inside a batch is legal; grouping by kind avoids subchannel switches.
			for (Section &section: sections_)
			{
				for (QueueSection &queueSection: section.queues)
				{
					for (CompiledBatch &batch: queueSection.batches)
					{
						std::stable_sort(batch.passes.begin(),
										 batch.passes.end(),
										 [this](PassId a, PassId b)
										 { return passes_[a.index].desc.kind < passes_[b.index].desc.kind; });
					}
				}
			}
		}
	}

	void RenderGraph::PlaceTransientMemory(bool alias)
	{
		std::vector<uint32_t> transients;
		for (uint32_t r = 0; r < resources_.size(); ++r)
		{
			ResourceNode &node = resources_[r];
			if (!OwnsTransientMemory(node))
			{
				continue;
			}
			std::tie(node.firstBatch, node.lastBatch) = ComputeLifetime(node);
			transients.push_back(r);
		}

		// Mips share their texture's placement, so the barrier code finds them like any other occupant.
		auto copyToMips = [this, &transients]()
		{
			for (uint32_t r: transients)
			{
				for (uint32_t m: resources_[r].mips)
				{
					ResourceNode &mipNode = resources_[m];
					mipNode.offset = resources_[r].offset;
					mipNode.size = resources_[r].size;
					mipNode.alignment = resources_[r].alignment;
					mipNode.firstBatch = resources_[r].firstBatch;
					mipNode.lastBatch = resources_[r].lastBatch;
				}
			}
		};

		transientMemorySize_ = 0;
		if (!alias)
		{
			for (uint32_t r: transients)
			{
				ResourceNode &node = resources_[r];
				node.offset = AlignUp(transientMemorySize_, node.alignment);
				transientMemorySize_ = node.offset + node.size;
			}
			copyToMips();
			return;
		}

		// Longest lifetime first (a stable base), then larger first; short-lived ones fill the gaps.
		std::sort(transients.begin(),
				  transients.end(),
				  [this](uint32_t a, uint32_t b)
				  {
					  const ResourceNode &na = resources_[a];
					  const ResourceNode &nb = resources_[b];
					  const uint32_t la = na.lastBatch - na.firstBatch;
					  const uint32_t lb = nb.lastBatch - nb.firstBatch;
					  if (la != lb)
						  return la > lb;
					  if (na.size != nb.size)
						  return na.size > nb.size;
					  return a < b;
				  });

		std::vector<uint32_t> placed;
		for (uint32_t r: transients)
		{
			ResourceNode &node = resources_[r];

			// Placed resources whose lifetime rectangles share a batch with this one, by offset.
			std::vector<uint32_t> conflicts;
			for (uint32_t other: placed)
			{
				const ResourceNode &o = resources_[other];
				if (o.firstBatch <= node.lastBatch && node.firstBatch <= o.lastBatch)
				{
					conflicts.push_back(other);
				}
			}
			std::sort(conflicts.begin(),
					  conflicts.end(),
					  [this](uint32_t a, uint32_t b) { return resources_[a].offset < resources_[b].offset; });

			// Lowest offset that fits between (or after) the conflicting ranges.
			uint64_t offset = 0;
			for (uint32_t other: conflicts)
			{
				const ResourceNode &o = resources_[other];
				if (AlignUp(offset, node.alignment) + node.size <= o.offset)
				{
					break;
				}
				offset = std::max(offset, o.offset + o.size);
			}
			node.offset = AlignUp(offset, node.alignment);
			transientMemorySize_ = std::max(transientMemorySize_, node.offset + node.size);
			placed.push_back(r);
		}
		copyToMips();
	}

	void RenderGraph::AddQueueWait(QueueType waitingQueue,
								   uint32_t waitingSection,
								   QueueType signalQueue,
								   uint32_t signalSection)
	{
		assert(waitingQueue != signalQueue);
		assert(signalSection < waitingSection);
		pendingWaits_.push_back({ waitingQueue, waitingSection, signalQueue, signalSection });
	}

	void RenderGraph::BuildBarriersAndSyncs()
	{
		const uint32_t epilogue = GetEpilogueSection();
		// Where each imported node is left at the end, on the graphics queue.
		std::vector<ResourceState> finalState(resources_.size());

		for (uint32_t r = 0; r < resources_.size(); ++r)
		{
			const ResourceNode &node = resources_[r];
			if (node.timeline.empty())
			{
				continue;
			}
			const ResourceId id{ r };
			const ResourceState released = ReleasedState(node.isTexture);

			ResourceState previousState; // state after the previous group's barrier
			for (uint32_t i = 0; i < node.timeline.size(); ++i)
			{
				const AccessGroup &group = groups_[node.timeline[i]];
				// As late as possible: right before the batch of the group's first pass.
				CompiledBatch &target =
						sections_[group.section].queues[QueueIndex(group.queue)].batches[group.minBatch];

				if (i == 0)
				{
					if (node.imported)
					{
						// Imported resources come from the graphics queue, in whatever state the context tracked.
						if (group.queue == QueueType::Graphics)
						{
							target.barriers.push_back({ id, BarrierKind::Transition, {}, group.state, true });
						}
						else
						{
							if (node.isTexture)
							{
								sections_[kPrologueSection].queues[QueueIndex(QueueType::Graphics)].releases.push_back(
										{ id, BarrierKind::QueueRelease, {}, released, true });
							}
							AddQueueWait(group.queue, group.section, QueueType::Graphics, kPrologueSection);
							target.barriers.push_back({ id, BarrierKind::QueueAcquire, released, group.state });
						}
					}
					else
					{
						// Transient: contents start undefined. The barrier must still wait for the resources
						// that used this memory before it.
						ResourceState before{ ResourceLayout::Undefined, ResourceSync::None, ResourceAccess::NoAccess };
						bool firstOccupant = true;
						for (uint32_t o = 0; o < resources_.size(); ++o)
						{
							const ResourceNode &other = resources_[o];
							if (o == r || other.imported || other.timeline.empty() ||
								other.lastBatch >= node.firstBatch ||
								!RangesOverlap(node.offset, node.size, other.offset, other.size))
							{
								continue;
							}
							firstOccupant = false;
							const AccessGroup &otherLast = groups_[other.timeline.back()];
							if (otherLast.queue == group.queue)
							{
								before.sync = before.sync | otherLast.state.sync;
							}
							else
							{
								AddQueueWait(group.queue, group.section, otherLast.queue, otherLast.section);
							}
						}
						if (firstOccupant)
						{
							// The previous execution of the graph may still be using this memory on this queue.
							before.sync = ResourceSync::All;
						}
						target.barriers.push_back({ id, BarrierKind::Initial, before, group.state });
					}
					previousState = group.state;
					continue;
				}

				const AccessGroup &previous = groups_[node.timeline[i - 1]];
				if (previous.queue == group.queue)
				{
					// Reads after reads need nothing as long as the earlier barrier already made the data visible
					// to every new access, in the same layout.
					const bool bothReads = !previous.isWrite && !group.isWrite;
					if (bothReads && previousState.layout == group.state.layout &&
						HasAll(previousState.sync, group.state.sync) &&
						HasAll(previousState.access, group.state.access))
					{
						continue;
					}
					target.barriers.push_back({ id, BarrierKind::Transition, previousState, group.state });
				}
				else
				{
					// Queue transfer: release textures to COMMON, wait on the producer's submit, acquire before use.
					// Buffers need no release since every submit boundary already flushes and syncs everything.
					if (node.isTexture)
					{
						sections_[previous.section].queues[QueueIndex(previous.queue)].releases.push_back(
								{ id, BarrierKind::QueueRelease, previousState, released });
					}
					AddQueueWait(group.queue, group.section, previous.queue, previous.section);
					target.barriers.push_back({ id, BarrierKind::QueueAcquire, released, group.state });
				}
				previousState = group.state;
			}

			// Imported resources go back to the graphics queue.
			const AccessGroup &last = groups_[node.timeline.back()];
			finalState[r] = previousState;
			if (node.imported && last.queue != QueueType::Graphics)
			{
				if (node.isTexture)
				{
					sections_[last.section].queues[QueueIndex(last.queue)].releases.push_back(
							{ id, BarrierKind::QueueRelease, previousState, released });
				}
				AddQueueWait(QueueType::Graphics, epilogue, last.queue, last.section);
				finalState[r] = released;
			}
		}

		// The context tracks one state per texture, so an import used by mip ends with all mips in one state:
		// their shared layout, or COMMON if they differ or a mip went unused (known only at Execute).
		for (uint32_t r = 0; r < resources_.size(); ++r)
		{
			const ResourceNode &node = resources_[r];
			if (!node.imported || node.mips.empty() || !IsUsed(node))
			{
				continue;
			}
			const uint32_t first = node.mips.front();
			bool allUsed = true;
			bool sameState = true;
			bool sameLayout = true;
			ResourceState merged{ finalState[first].layout,
								  ResourceSync::None,
								  ResourceAccess::Common }; // Common is no bits
			for (uint32_t m: node.mips)
			{
				allUsed = allUsed && !resources_[m].timeline.empty();
				sameState = sameState && SameState(finalState[m], finalState[first]);
				sameLayout = sameLayout && finalState[m].layout == finalState[first].layout;
				merged.sync = merged.sync | finalState[m].sync;
				merged.access = merged.access | finalState[m].access;
			}
			if (allUsed && sameState)
			{
				continue;
			}
			// The next barrier in this command list chains on this one, so it needs a real sync scope and an
			// access its layout allows (NO_ACCESS with a sync scope is only valid for UNDEFINED).
			const ResourceState target = allUsed && sameLayout && merged.sync != ResourceSync::None
												 ? merged
												 : ResourceState{ ResourceLayout::Common,
																  ResourceSync::AllShading,
																  ResourceAccess::ShaderResource };
			auto &releases = sections_[epilogue].queues[QueueIndex(QueueType::Graphics)].releases;
			for (uint32_t m: node.mips)
			{
				if (resources_[m].timeline.empty())
				{
					releases.push_back({ ResourceId{ m }, BarrierKind::Transition, {}, target, true });
				}
				else if (!SameState(finalState[m], target))
				{
					releases.push_back({ ResourceId{ m }, BarrierKind::Transition, finalState[m], target });
				}
			}
		}
	}

	void RenderGraph::FinaliseQueueWaits()
	{
		// In section order. A wait is redundant if the waiting queue has already waited for the same or a
		// later submit of the signalling queue.
		std::sort(pendingWaits_.begin(),
				  pendingWaits_.end(),
				  [](const QueueWaitInfo &a, const QueueWaitInfo &b)
				  {
					  if (a.waitingSection != b.waitingSection)
						  return a.waitingSection < b.waitingSection;
					  if (a.waitingQueue != b.waitingQueue)
						  return a.waitingQueue < b.waitingQueue;
					  if (a.signalQueue != b.signalQueue)
						  return a.signalQueue < b.signalQueue;
					  return a.signalSection > b.signalSection;
				  });

		int64_t waited[kQueueTypeCount][kQueueTypeCount];
		for (auto &row: waited)
		{
			row[0] = row[1] = -1;
		}

		for (const QueueWaitInfo &wait: pendingWaits_)
		{
			int64_t &already = waited[QueueIndex(wait.waitingQueue)][QueueIndex(wait.signalQueue)];
			if (static_cast<int64_t>(wait.signalSection) <= already)
			{
				continue;
			}
			already = wait.signalSection;
			queueWaits_.push_back(wait);
			sections_[wait.waitingSection].queues[QueueIndex(wait.waitingQueue)].waits.push_back(wait);
			sections_[wait.signalSection].queues[QueueIndex(wait.signalQueue)].signal = true;
		}
		pendingWaits_.clear();

		for (Section &section: sections_)
		{
			for (QueueSection &queueSection: section.queues)
			{
				queueSection.active = !queueSection.batches.empty() || !queueSection.releases.empty() ||
									  !queueSection.waits.empty() || queueSection.signal;
			}
			usesComputeQueue_ = usesComputeQueue_ || section.queues[QueueIndex(QueueType::Compute)].active;
		}
	}

	// Execute

	void RenderGraph::CreateTransientResources(RenderContext &rc)
	{
		if (transientMemorySize_ > 0)
		{
			heap_ = rc.CreateMemoryHeap(transientMemorySize_, "RenderGraphTransientHeap");
		}
		for (ResourceNode &node: resources_)
		{
			if (!OwnsTransientMemory(node))
			{
				continue;
			}
			if (node.isTexture)
			{
				TextureDesc desc = node.textureDesc;
				desc.debugName = node.name;
				node.handle = rc.CreatePlacedTexture(heap_, node.offset, desc, node.viewType);
			}
			else
			{
				BufferDesc desc = node.bufferDesc;
				desc.debugName = node.name;
				node.handle = rc.CreatePlacedBuffer(heap_, node.offset, desc, node.viewType);
			}
		}
		realisedContext_ = &rc;
	}

	void RenderGraph::ReleaseResources()
	{
		if (!realisedContext_)
		{
			return;
		}
		// Views before the textures they view.
		for (ResourceNode &node: resources_)
		{
			for (const TextureViews &view: node.views)
			{
				if (view.srv != InvalidResourceHandle)
					realisedContext_->DestroyResource(view.srv);
				if (view.uav != InvalidResourceHandle)
					realisedContext_->DestroyResource(view.uav);
			}
			node.views.clear();
		}
		for (ResourceNode &node: resources_)
		{
			if (!node.imported && node.handle != InvalidResourceHandle)
			{
				realisedContext_->DestroyResource(node.handle);
				node.handle = InvalidResourceHandle;
			}
		}
		if (heap_ != InvalidMemoryHeapHandle)
		{
			realisedContext_->DestroyMemoryHeap(heap_);
			heap_ = InvalidMemoryHeapHandle;
		}
		realisedContext_ = nullptr;
	}

	ResourceTransitionDesc RenderGraph::ResolveBarrier(const BarrierInfo &barrier) const
	{
		const ResourceNode &node = resources_[barrier.resource.index];
		const ResourceNode &owner = resources_[Owner(barrier.resource.index)];
		const ResourceState before = barrier.beforeFromTrackedState ? owner.trackedState : barrier.before;
		return ResourceTransitionDesc{ owner.handle,  node.parent != kNoNode ? node.mip : UINT32_MAX,
									   before.layout, barrier.after.layout,
									   before.sync,	  barrier.after.sync,
									   before.access, barrier.after.access };
	}

	void RenderGraph::RecordBarrierList(RenderContext &rc,
										std::span<const BarrierInfo> barriers,
										QueueType queue,
										uint32_t section)
	{
		if (barriers.empty())
		{
			return;
		}
		std::vector<ResourceTransitionDesc> descs;
		descs.reserve(barriers.size());
		for (const BarrierInfo &barrier: barriers)
		{
			descs.push_back(ResolveBarrier(barrier));
		}
		rc.RecordBarriers(descs);

		ExecutionEvent event;
		event.type = ExecutionEventType::Barriers;
		event.queue = queue;
		event.section = section;
		event.barrierCount = static_cast<uint32_t>(barriers.size());
		trace_.push_back(event);
	}

	void RenderGraph::Execute(RenderContext &rc)
	{
		if (!compiled_)
		{
			throw std::logic_error("RenderGraph::Execute: the graph is not compiled");
		}
		if (!rc.IsQueueCommandListOpen(QueueType::Graphics))
		{
			throw std::logic_error("RenderGraph::Execute: the graphics command list must be open");
		}
		if (!realisedContext_)
		{
			CreateTransientResources(rc);
		}
		assert(realisedContext_ == &rc);

		// Read once up front, since mip barriers update the one tracked state as they go.
		for (ResourceNode &node: resources_)
		{
			if (node.imported && node.parent == kNoNode && IsUsed(node))
			{
				node.trackedState = rc.GetResourceState(node.handle);
			}
		}

		trace_.clear();
		auto traceEvent = [this](ExecutionEventType type, QueueType queue, uint32_t section)
		{
			ExecutionEvent event;
			event.type = type;
			event.queue = queue;
			event.section = section;
			trace_.push_back(event);
			return &trace_.back();
		};

		// The graphics list may already hold the frame's earlier commands.
		bool dirty[kQueueTypeCount] = { true, false };
		std::vector<std::array<QueueSyncPoint, kQueueTypeCount>> signals(sections_.size());

		if (usesComputeQueue_)
		{
			// Transient memory, and the queue each imported resource is on, carry over between executions.
			// Each queue waits for the other's work from the previous one; that is usually long finished.
			const QueueSyncPoint lastGraphics = rc.GetLastSyncPoint(QueueType::Graphics);
			const QueueSyncPoint lastCompute = rc.GetLastSyncPoint(QueueType::Compute);
			rc.QueueWait(QueueType::Compute, lastGraphics);
			traceEvent(ExecutionEventType::FrameStartWait, QueueType::Compute, kPrologueSection)->otherQueue =
					QueueType::Graphics;
			rc.QueueWait(QueueType::Graphics, lastCompute);
			traceEvent(ExecutionEventType::FrameStartWait, QueueType::Graphics, kPrologueSection)->otherQueue =
					QueueType::Compute;
		}

		for (uint32_t s = 0; s < sections_.size(); ++s)
		{
			for (uint32_t q = 0; q < kQueueTypeCount; ++q)
			{
				const QueueType queue = static_cast<QueueType>(q);
				const QueueSection &queueSection = sections_[s].queues[q];
				if (!queueSection.active)
				{
					continue;
				}

				if (!queueSection.waits.empty())
				{
					// A queue wait holds back everything submitted after it, so submit earlier work first.
					if (rc.IsQueueCommandListOpen(queue) && dirty[q])
					{
						rc.SubmitQueue(queue);
						dirty[q] = false;
						traceEvent(ExecutionEventType::Submit, queue, s);
					}
					for (const QueueWaitInfo &wait: queueSection.waits)
					{
						const QueueSyncPoint point = signals[wait.signalSection][QueueIndex(wait.signalQueue)];
						assert(point.value != 0 && "waiting on a submit that has not happened");
						rc.QueueWait(queue, point);
						ExecutionEvent *event = traceEvent(ExecutionEventType::Wait, queue, s);
						event->otherQueue = wait.signalQueue;
						event->otherSection = wait.signalSection;
					}
				}

				if (!rc.IsQueueCommandListOpen(queue))
				{
					rc.BeginQueueCommandList(queue);
				}
				rc.SetRecordingQueue(queue);

				for (const CompiledBatch &batch: queueSection.batches)
				{
					RecordBarrierList(rc, batch.barriers, queue, s);
					for (PassId passId: batch.passes)
					{
						const PassNode &pass = passes_[passId.index];
						rc.BeginEvent(pass.desc.name);
						if (pass.desc.execute)
						{
							PassContext context(rc, *this, passId);
							pass.desc.execute(context);
							rc.SetRecordingQueue(queue);
						}
						rc.EndEvent();
						traceEvent(ExecutionEventType::Pass, queue, s)->pass = passId;
					}
					dirty[q] = true;
				}

				RecordBarrierList(rc, queueSection.releases, queue, s);
				dirty[q] = dirty[q] || !queueSection.releases.empty();

				if (queueSection.signal)
				{
					signals[s][q] = rc.SubmitQueue(queue);
					dirty[q] = false;
					traceEvent(ExecutionEventType::Submit, queue, s);
				}
			}
		}

		if (rc.IsQueueCommandListOpen(QueueType::Compute))
		{
			rc.SubmitQueue(QueueType::Compute);
			traceEvent(ExecutionEventType::Submit, QueueType::Compute, GetEpilogueSection());
		}
		if (!rc.IsQueueCommandListOpen(QueueType::Graphics))
		{
			rc.BeginQueueCommandList(QueueType::Graphics);
		}
		rc.SetRecordingQueue(QueueType::Graphics);
	}

	// Introspection

	uint32_t RenderGraph::GetBatchCount(uint32_t section, QueueType queue) const
	{
		assert(section < sections_.size());
		return static_cast<uint32_t>(sections_[section].queues[QueueIndex(queue)].batches.size());
	}

	uint32_t RenderGraph::GetTotalBatchCount() const
	{
		uint32_t total = 0;
		for (const Section &section: sections_)
		{
			for (const QueueSection &queueSection: section.queues)
			{
				total += static_cast<uint32_t>(queueSection.batches.size());
			}
		}
		return total;
	}

	const CompiledBatch &RenderGraph::GetBatch(uint32_t section, QueueType queue, uint32_t index) const
	{
		assert(section < sections_.size());
		const auto &batches = sections_[section].queues[QueueIndex(queue)].batches;
		assert(index < batches.size());
		return batches[index];
	}

	std::span<const BarrierInfo> RenderGraph::GetQueueReleases(uint32_t section, QueueType queue) const
	{
		assert(section < sections_.size());
		return sections_[section].queues[QueueIndex(queue)].releases;
	}

	bool RenderGraph::IsSubmitSignalled(uint32_t section, QueueType queue) const
	{
		assert(section < sections_.size());
		return sections_[section].queues[QueueIndex(queue)].signal;
	}

	uint32_t RenderGraph::GetPassBatch(PassId pass) const
	{
		assert(pass.index < passes_.size());
		return passes_[pass.index].batch;
	}

	uint32_t RenderGraph::GetPassSection(PassId pass) const
	{
		assert(pass.index < passes_.size());
		return passes_[pass.index].section;
	}

	QueueType RenderGraph::GetPassQueue(PassId pass) const
	{
		assert(pass.index < passes_.size());
		return passes_[pass.index].desc.queue;
	}

	const std::string &RenderGraph::GetPassName(PassId pass) const
	{
		assert(pass.index < passes_.size());
		return passes_[pass.index].desc.name;
	}

	bool RenderGraph::PassUses(PassId pass, ResourceId resource) const
	{
		assert(pass.index < passes_.size());
		// Using a whole texture uses each of its mips.
		const uint32_t parent = resources_[resource.index].parent;
		for (const Attachment &attachment: passes_[pass.index].desc.attachments)
		{
			if (attachment.resource == resource || (parent != kNoNode && attachment.resource.index == parent))
			{
				return true;
			}
		}
		return false;
	}

	Access RenderGraph::GetPassAccess(PassId pass, ResourceId resource) const
	{
		const uint32_t parent = resources_[resource.index].parent;
		for (const Attachment &attachment: passes_[pass.index].desc.attachments)
		{
			if (attachment.resource == resource || (parent != kNoNode && attachment.resource.index == parent))
			{
				return attachment.access;
			}
		}
		assert(false && "the pass does not use the resource");
		return Access::ComputeShaderRead;
	}

	ResourceHandle RenderGraph::GetView(RenderContext &rc, uint32_t resource, bool unorderedAccess) const
	{
		const ResourceNode &node = resources_[resource];
		const bool isMip = node.parent != kNoNode;
		const ResourceHandle texture = resources_[Owner(resource)].handle;
		assert(texture != InvalidResourceHandle);
		auto it = std::find_if(node.views.begin(),
							   node.views.end(),
							   [&](const TextureViews &v) { return v.texture == texture; });
		if (it == node.views.end())
		{
			node.views.push_back({ texture });
			it = node.views.end() - 1;
		}
		ResourceHandle &view = unorderedAccess ? it->uav : it->srv;
		if (view == InvalidResourceHandle)
		{
			const TextureDesc desc = rc.GetTextureDesc(texture);
			TextureViewDesc viewDesc;
			viewDesc.viewType = unorderedAccess ? ResourceViewType::UnorderedAccess : ResourceViewType::ShaderResource;
			viewDesc.dimension = desc.dimension;
			viewDesc.asArray = desc.dimension != TextureDimension::Texture3D && desc.depthOrArraySize > 1;
			viewDesc.baseMip = isMip ? node.mip : 0; // a whole texture's UAV is of mip 0
			viewDesc.mipCount = isMip ? 1 : 0;
			view = rc.CreateTextureView(texture, viewDesc);
		}
		return view;
	}

	std::optional<TransientAllocation> RenderGraph::GetTransientAllocation(ResourceId resource) const
	{
		assert(resource.index < resources_.size());
		const ResourceNode &node = resources_[resource.index];
		if (node.imported || !IsUsed(node) || !compiled_)
		{
			return std::nullopt;
		}
		return TransientAllocation{ node.offset, node.size, node.alignment, node.firstBatch, node.lastBatch };
	}

	uint64_t RenderGraph::GetVirtualArea() const
	{
		uint64_t area = 0;
		for (const ResourceNode &node: resources_)
		{
			area += ResourceArea(node);
		}
		return area;
	}

	ResourceHandle RenderGraph::GetHandle(ResourceId resource) const
	{
		assert(resource.index < resources_.size());
		return resources_[Owner(resource.index)].handle;
	}

	std::string RenderGraph::Dump() const
	{
		std::ostringstream out;
		char hex[32];
		auto state = [&hex](const ResourceState &s)
		{
			std::snprintf(hex,
						  sizeof(hex),
						  "L%d/S%x/A%x",
						  static_cast<int>(s.layout),
						  static_cast<uint32_t>(s.sync),
						  static_cast<uint32_t>(s.access));
			return std::string(hex);
		};
		auto barrierText = [&](const BarrierInfo &b)
		{
			return resources_[b.resource.index].name + "(" + ToString(b.kind) + " " +
				   (b.beforeFromTrackedState ? std::string("tracked") : state(b.before)) + " -> " + state(b.after) +
				   ")";
		};

		for (uint32_t s = 0; s < sections_.size(); ++s)
		{
			const Section &section = sections_[s];
			out << "section " << s
				<< (s == kPrologueSection		? " (prologue)"
					: s == GetEpilogueSection() ? " (epilogue)"
												: "")
				<< " start=" << section.globalStart << " length=" << section.length << "\n";
			for (uint32_t q = 0; q < kQueueTypeCount; ++q)
			{
				const QueueSection &queueSection = section.queues[q];
				if (!queueSection.active)
				{
					continue;
				}
				out << "  [" << ToString(static_cast<QueueType>(q)) << "]" << (queueSection.signal ? " signals" : "")
					<< "\n";
				for (const QueueWaitInfo &wait: queueSection.waits)
				{
					out << "    wait " << ToString(wait.signalQueue) << "@" << wait.signalSection << "\n";
				}
				for (const CompiledBatch &batch: queueSection.batches)
				{
					out << "    batch " << batch.index << ":";
					for (const BarrierInfo &barrier: batch.barriers)
					{
						out << " " << barrierText(barrier);
					}
					out << " |";
					for (PassId pass: batch.passes)
					{
						out << " " << passes_[pass.index].desc.name;
					}
					out << "\n";
				}
				for (const BarrierInfo &barrier: queueSection.releases)
				{
					out << "    end: " << barrierText(barrier) << "\n";
				}
			}
		}
		out << "transient memory: " << transientMemorySize_ << " bytes\n";
		for (const ResourceNode &node: resources_)
		{
			if (OwnsTransientMemory(node))
			{
				out << "  " << node.name << " offset=" << node.offset << " size=" << node.size << " batches=["
					<< node.firstBatch << "," << node.lastBatch << "]\n";
			}
		}
		return out.str();
	}
} // namespace rg
