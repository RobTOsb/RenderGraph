// Per-frame constants shared by every pass; matches FrameConstants in main.cpp.
struct FrameConstants
{
	float4x4 viewProj;
	float4x4 invViewProj;
	float3 cameraPos;
	float ssaoRadius;
	float3 lightPos;
	float lightIntensity;
	float2 size;
	float nearZ;
	float farZ;
	uint debugView; // 0 lit, 1 AO, 2 albedo, 3 normals
	uint applyGamma; // 1 when the back buffer is not sRGB
	uint useAO;
	float rtaoRadius;
};

ConstantBuffer<FrameConstants> g_Frame;

[shader("vertex")]
float4 vsFullscreen(uint vertexId: SV_VertexID) : SV_Position
{
	const float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
	return float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
}

float3 ReconstructWorld(float2 pixel, float depth)
{
	const float2 ndc = float2(pixel.x / g_Frame.size.x * 2.0f - 1.0f, 1.0f - pixel.y / g_Frame.size.y * 2.0f);
	const float4 world = mul(float4(ndc, depth, 1.0f), g_Frame.invViewProj);
	return world.xyz / world.w;
}

// View distance from a reversed Z depth value.
float LinearDepth(float depth)
{
	return g_Frame.farZ * g_Frame.nearZ / (g_Frame.nearZ + depth * (g_Frame.farZ - g_Frame.nearZ));
}
