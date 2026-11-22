// Deferred lighting: one downward facing ceiling light, ambient scaled by SSAO, Reinhard tone mapping.
import common;

Texture2D<float4> g_Albedo;
Texture2D<float4> g_Normal;
Texture2D<float> g_Depth;
Texture2D<float> g_AO;

float3 Output(float3 colour)
{
	return g_Frame.applyGamma != 0 ? pow(saturate(colour), 1.0f / 2.2f) : saturate(colour);
}

[shader("vertex")]
float4 vsFullscreen(uint vertexId: SV_VertexID) : SV_Position
{
	return FullscreenPosition(vertexId);
}

[shader("fragment")]
float4 psLighting(float4 position: SV_Position) : SV_Target
{
	const int3 pixel = int3(int2(position.xy), 0);
	const float depth = g_Depth.Load(pixel);
	if (!(depth > 0.0f))
	{
		return float4(0.0f, 0.0f, 0.0f, 1.0f); // background, or a NaN depth
	}

	const float4 albedoEmissive = g_Albedo.Load(pixel);
	const float3 albedo = albedoEmissive.rgb;
	// A zero normal means no surface was written here; normalizing it would give NaN.
	const float3 rawNormal = g_Normal.Load(pixel).xyz;
	const float normalLengthSq = dot(rawNormal, rawNormal);
	if (!(normalLengthSq > 1e-8f))
	{
		return float4(0.0f, 0.0f, 0.0f, 1.0f);
	}
	const float3 n = rawNormal * rsqrt(normalLengthSq);
	const float ao = g_Frame.useAO != 0 ? g_AO.Load(pixel) : 1.0f;

	switch (g_Frame.debugView)
	{
		case 1: return float4(ao.xxx, 1.0f);
		case 2: return float4(Output(albedo), 1.0f);
		case 3: return float4(n * 0.5f + 0.5f, 1.0f);
		default: break;
	}

	if (albedoEmissive.a > 0.0f)
	{
		return float4(Output(albedo), 1.0f); // the light itself
	}

	const float3 p = ReconstructWorld(position.xy, depth);
	const float3 toLight = g_Frame.lightPos - p;
	const float distanceSq = dot(toLight, toLight);
	const float3 l = toLight * rsqrt(distanceSq);
	// The light faces down, so it emits with a cosine falloff and leaves the ceiling unlit.
	const float emission = saturate(l.y);
	const float3 direct = albedo * saturate(dot(n, l)) * emission * g_Frame.lightIntensity / (1.0f + distanceSq);
	// Stand-in for bounce light, which is what ambient occlusion darkens.
	const float3 ambient = albedo * 0.35f;

	float3 colour = ambient * ao + direct * lerp(1.0f, ao, 0.75f);
	colour = colour / (1.0f + colour);
	return float4(Output(colour), 1.0f);
}
