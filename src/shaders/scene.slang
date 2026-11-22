// Cornell box geometry: the prepass uses psPrepass and the G-buffer pass psGBuffer.
// Both pipelines share vsMain, so the prepass depth matches exactly for the EQUAL depth test.
#include "common.slang"

struct Vertex
{
	float3 position;
	float3 normal;
	float3 albedo;
	float emissive;
};

StructuredBuffer<Vertex> g_Vertices;

struct VSOut
{
	float4 position : SV_Position;
	float3 normal : NORMAL;
	float3 albedo : COLOR0;
	float emissive : COLOR1;
};

[shader("vertex")]
VSOut vsMain(uint vertexId: SV_VertexID)
{
	const Vertex v = g_Vertices[vertexId];
	VSOut o;
	o.position = mul(float4(v.position, 1.0f), g_Frame.viewProj);
	o.normal = v.normal;
	o.albedo = v.albedo;
	o.emissive = v.emissive;
	return o;
}

// What SSAO reads, written by the prepass so SSAO on the compute queue shares nothing with the G-buffer pass.
[shader("fragment")]
float4 psPrepass(VSOut i) : SV_Target
{
	return float4(normalize(i.normal), i.position.z); // SV_Position.z is the depth the prepass writes
}

struct GBufferOut
{
	float4 normal : SV_Target0; // world space normal
	float4 albedo : SV_Target1; // rgb albedo, a emissive
};

[shader("fragment")]
GBufferOut psGBuffer(VSOut i)
{
	GBufferOut o;
	o.albedo = float4(i.albedo, i.emissive);
	o.normal = float4(normalize(i.normal), 0.0f);
	return o;
}
