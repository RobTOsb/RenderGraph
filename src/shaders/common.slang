// Shared by every shader through import common; only public declarations are visible to them.
module common;

// Per-frame constants shared by every pass; matches FrameConstants in main.cpp.
public struct FrameConstants
{
	public float4x4 viewProj;
	public float4x4 invViewProj;
	public float3 cameraPos;
	public float ssaoRadius;
	public float3 lightPos;
	public float lightIntensity;
	public float2 size;
	public float nearZ;
	public float farZ;
	public uint debugView; // 0 lit, 1 AO, 2 albedo, 3 normals
	public uint applyGamma; // 1 when the back buffer is not sRGB
	public uint useAO;
	public float rtaoRadius;
};

public ConstantBuffer<FrameConstants> g_Frame;

// Clip space position of a full screen triangle vertex. Entry points must live in the loaded module, so each
// shader that draws full screen declares its own vertex entry point around this.
public float4 FullscreenPosition(uint vertexId)
{
	const float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
	return float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
}

public float3 ReconstructWorld(float2 pixel, float depth)
{
	const float2 ndc = float2(pixel.x / g_Frame.size.x * 2.0f - 1.0f, 1.0f - pixel.y / g_Frame.size.y * 2.0f);
	const float4 world = mul(float4(ndc, depth, 1.0f), g_Frame.invViewProj);
	return world.xyz / world.w;
}

// View distance from a reversed Z depth value.
public float LinearDepth(float depth)
{
	return g_Frame.farZ * g_Frame.nearZ / (g_Frame.nearZ + depth * (g_Frame.farZ - g_Frame.nearZ));
}
