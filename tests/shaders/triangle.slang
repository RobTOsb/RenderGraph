// The ray tracing test's triangle: (0,0,0), (s,0,0), (0,s,0) as float3 positions, and its indices.

RWStructuredBuffer<uint> Vertices;
RWStructuredBuffer<uint> Indices;

[numthreads(1, 1, 1)]
void main()
{
	const float s = 1.01f;
	const float positions[9] = { 0.0f, 0.0f, 0.0f, s, 0.0f, 0.0f, 0.0f, s, 0.0f };
	for (uint i = 0; i < 9; ++i)
	{
		Vertices[i] = asuint(positions[i]);
	}
	for (uint i = 0; i < 3; ++i)
	{
		Indices[i] = i;
	}
}
