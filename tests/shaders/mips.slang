// Bloom style mip chain on R32_UINT textures: each entry point works on one mip, bound through a single mip view.
struct PushConstants
{
	uint value;
	uint width; // of the mip being written
	uint height;
	uint offset;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> g_PushConstants;

Texture2D<uint> Input : register(t0);
RWTexture2D<uint> Output : register(u0);
RWStructuredBuffer<uint> Buffer : register(u1);

bool Outside(uint2 p)
{
	return p.x >= g_PushConstants.width || p.y >= g_PushConstants.height;
}

// Output = value
[shader("compute")]
[numthreads(8, 8, 1)]
void seed(uint3 id: SV_DispatchThreadID)
{
	if (!Outside(id.xy))
		Output[id.xy] = g_PushConstants.value;
}

// Output = sum of the 2x2 texels of the finer mip
[shader("compute")]
[numthreads(8, 8, 1)]
void down(uint3 id: SV_DispatchThreadID)
{
	if (Outside(id.xy))
		return;
	const int2 p = int2(id.xy) * 2;
	Output[id.xy] = Input.Load(int3(p, 0)) + Input.Load(int3(p + int2(1, 0), 0)) + Input.Load(int3(p + int2(0, 1), 0)) +
					Input.Load(int3(p + int2(1, 1), 0));
}

// Output += the coarser mip's texel
[shader("compute")]
[numthreads(8, 8, 1)]
void up(uint3 id: SV_DispatchThreadID)
{
	if (!Outside(id.xy))
		Output[id.xy] = Output[id.xy] + Input.Load(int3(id.xy / 2, 0));
}

// Buffer[offset + y * width + x] = Input
[shader("compute")]
[numthreads(8, 8, 1)]
void readback(uint3 id: SV_DispatchThreadID)
{
	if (!Outside(id.xy))
		Buffer[g_PushConstants.offset + id.y * g_PushConstants.width + id.x] = Input.Load(int3(id.xy, 0));
}

// Raster version of down: one fullscreen triangle over the mip being drawn.
[shader("vertex")]
float4 vsFullscreen(uint vertexId: SV_VertexID) : SV_Position
{
	const float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
	return float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
}

[shader("fragment")]
uint psDown(float4 position: SV_Position) : SV_Target
{
	const int2 p = int2(position.xy) * 2;
	return Input.Load(int3(p, 0)) + Input.Load(int3(p + int2(1, 0), 0)) + Input.Load(int3(p + int2(0, 1), 0)) +
		   Input.Load(int3(p + int2(1, 1), 0));
}
