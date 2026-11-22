// Amplification, mesh and pixel shaders all read Input; the mesh and pixel shaders write a buffer each.
// One fullscreen triangle over an 8x8 viewport gives one pixel shader invocation per element.

StructuredBuffer<uint> Input;
RWStructuredBuffer<uint> MeshOut;
RWStructuredBuffer<uint> PixelOut;

struct Params
{
	uint scale;
};
ConstantBuffer<Params> g_Params;

struct Payload
{
	uint value;
};
groupshared Payload s_payload;

[shader("amplification")]
[numthreads(1, 1, 1)]
void asMain()
{
	s_payload.value = Input[0];
	DispatchMesh(1, 1, 1, s_payload);
}

struct VertexOut
{
	float4 position : SV_Position;
};

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void msMain(uint tid: SV_GroupThreadID, in payload Payload p, out vertices VertexOut verts[3], out indices uint3 tris[1])
{
	SetMeshOutputCounts(3, 1);
	MeshOut[tid] = Input[tid] + p.value * g_Params.scale;
	if (tid < 3)
	{
		const float2 uv = float2((tid << 1) & 2, tid & 2);
		verts[tid].position = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
	}
	if (tid == 0)
	{
		tris[0] = uint3(0, 1, 2);
	}
}

[shader("fragment")]
void psMain(float4 position: SV_Position)
{
	const uint index = uint(position.y) * 8 + uint(position.x);
	PixelOut[index] = Input[index] * 2 + 1;
}
