// World space SSAO with a 4x4 interleaved rotation pattern, then a depth aware 4x4 blur that removes it.
// Both run on the async compute queue and read only the prepass's normal and depth, so they overlap the G-buffer.
// Checks are written as !(x > limit) so NaN inputs fail them too, and both passes never output NaN.
#include "common.slang"

Texture2D<float4> g_Input; // xyz world space normal, w reversed Z depth (0 where nothing was drawn)
Texture2D<float> g_AO;
RWTexture2D<float> g_Out;

static const uint kSampleCount = 16;
static const float kPi = 3.14159265f;

bool Outside(int2 pixel)
{
	return any(pixel >= int2(g_Frame.size));
}

float Depth(int2 pixel)
{
	return g_Input.Load(int3(pixel, 0)).w;
}

float ComputeAO(int2 pixel)
{
	const float4 input = g_Input.Load(int3(pixel, 0));
	const float depth = input.w;
	if (!(depth > 0.0f))
	{
		return 1.0f; // background
	}

	// A zero normal means no surface was written here; normalizing it would give NaN.
	const float normalLengthSq = dot(input.xyz, input.xyz);
	if (!(normalLengthSq > 1e-8f))
	{
		return 1.0f;
	}
	const float3 n = input.xyz * rsqrt(normalLengthSq);
	const float3 p = ReconstructWorld(float2(pixel) + 0.5f, depth);
	const float3 helper = abs(n.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
	const float3 t = normalize(cross(helper, n));
	const float3 b = cross(n, t);

	// Same rotation for every pixel at one position in a 4x4 tile, so the blur averages all 16.
	const uint tile = (uint(pixel.x) & 3u) + (uint(pixel.y) & 3u) * 4u;
	const float rotation = (float(tile) + 0.5f) / 16.0f * 2.0f * kPi;
	const float cosR = cos(rotation);
	const float sinR = sin(rotation);
	const float radius = g_Frame.ssaoRadius;
	const float pDistance = length(p - g_Frame.cameraPos);

	float occlusion = 0.0f;
	for (uint i = 0; i < kSampleCount; ++i)
	{
		// Cosine weighted hemisphere direction from a Hammersley point, rotated per pixel.
		const float u1 = (float(i) + 0.5f) / float(kSampleCount);
		const float u2 = float(reversebits(i)) * 2.3283064365386963e-10f;
		const float r = sqrt(u1);
		const float phi = 2.0f * kPi * u2;
		const float2 disk = float2(r * cos(phi), r * sin(phi));
		const float2 rotated = float2(disk.x * cosR - disk.y * sinR, disk.x * sinR + disk.y * cosR);
		const float3 direction = t * rotated.x + b * rotated.y + n * sqrt(max(0.0f, 1.0f - u1));
		const float3 s = p + direction * radius * lerp(0.15f, 1.0f, u1 * u1);

		const float4 clip = mul(float4(s, 1.0f), g_Frame.viewProj);
		if (!(clip.w > 1e-6f))
		{
			continue; // behind the camera
		}
		const float2 ndc = clip.xy / clip.w;
		const float2 samplePixel = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f) * g_Frame.size;
		if (!(all(samplePixel >= 0.0f) && all(samplePixel < g_Frame.size)))
		{
			continue; // off screen
		}

		// Rebuild the occluder at the centre of the texel its depth came from, not at the sample position.
		const float2 texel = floor(samplePixel);
		const float sceneDepth = Depth(int2(texel));
		if (!(sceneDepth > 0.0f))
		{
			continue; // background never occludes
		}
		const float3 q = ReconstructWorld(texel + 0.5f, sceneDepth);
		const float sDistance = length(s - g_Frame.cameraPos);
		const float qDistance = length(q - g_Frame.cameraPos);
		// Only geometry above p's tangent plane occludes, so flat surfaces never occlude themselves.
		const bool aboveSurface = dot(q - p, n) > 0.02f * radius;
		if (aboveSurface && qDistance < sDistance - 0.01f)
		{
			// Fade out occluders far outside the radius, which are depth discontinuities.
			occlusion += smoothstep(0.0f, 1.0f, radius / max(abs(pDistance - qDistance), 1e-4f));
		}
	}
	const float ao = pow(saturate(1.0f - occlusion / float(kSampleCount)), 1.5f);
	return isnan(ao) ? 1.0f : ao;
}

float Blur(int2 pixel)
{
	const float centreRawDepth = Depth(pixel);
	if (!(centreRawDepth > 0.0f))
	{
		return 1.0f; // background
	}
	const float centreDepth = LinearDepth(centreRawDepth);

	float sum = 0.0f;
	float weightSum = 0.0f;
	for (int y = -2; y < 2; ++y)
	{
		for (int x = -2; x < 2; ++x)
		{
			const int2 tap = clamp(pixel + int2(x, y), int2(0, 0), int2(g_Frame.size) - 1);
			const float tapRawDepth = Depth(tap);
			const float tapAO = g_AO.Load(int3(tap, 0));
			if (!(tapRawDepth > 0.0f) || isnan(tapAO))
			{
				continue; // background or a bad value
			}
			// Taps within 5% of the centre depth count fully, so slanted surfaces still blur.
			const float tapDepth = LinearDepth(tapRawDepth);
			const float weight = saturate(1.0f - abs(tapDepth - centreDepth) / centreDepth * 20.0f) + 1e-4f;
			sum += tapAO * weight;
			weightSum += weight;
		}
	}
	return weightSum > 0.0f ? sum / weightSum : 1.0f;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csSSAO(uint3 id: SV_DispatchThreadID)
{
	if (!Outside(int2(id.xy)))
		g_Out[id.xy] = ComputeAO(int2(id.xy));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csBlur(uint3 id: SV_DispatchThreadID)
{
	if (!Outside(int2(id.xy)))
		g_Out[id.xy] = Blur(int2(id.xy));
}
