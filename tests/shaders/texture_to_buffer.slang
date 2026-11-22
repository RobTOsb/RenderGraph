// Output[y * width + x] = RGBA8 packing of InputTexture[x, y]
struct PushConstants
{
	uint width;
	uint height;
	uint2 pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> g_PushConstants;

Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
	if (id.x >= g_PushConstants.width || id.y >= g_PushConstants.height)
	{
		return;
	}
	const uint4 c = uint4(round(saturate(InputTexture.Load(int3(id.xy, 0))) * 255.0f));
	Output[id.y * g_PushConstants.width + id.x] = c.r | (c.g << 8) | (c.b << 16) | (c.a << 24);
}
