#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>


__device__ int getIndex(int textureSize_x, int textureSize_y, int textureCoord_x, int textureCoord_y, int axis)
{
	int index = 2 * (textureSize_x * textureCoord_y + textureCoord_x) + axis;
	bool isValid = textureCoord_x >= 0 && textureCoord_x < textureSize_x&& textureCoord_y >= 0 && textureCoord_y < textureSize_y;
	index = isValid * index + (1 - isValid) * -1;
	return index;
}


__global__ void calculateCoords(double* coords, int textureSize_x, int textureSize_y, double scale_x, double scale_y)
{
	int textureCoord_x = blockIdx.x * blockDim.x + threadIdx.x;
	int textureCoord_y = blockIdx.y * blockDim.y + threadIdx.y;

	int index_x = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y, 0);
	int index_y = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y, 1);

	if (index_x != -1 && index_y != -1) // make sure the indices are valid
	{
		double flatCoord_x = ((double)textureCoord_x / textureSize_x * 2.0 - 1.0) * scale_x;
		double flatCoord_y = ((double)textureCoord_y / textureSize_y * 2.0 - 1.0) * scale_y;

		double distance = sqrt(flatCoord_x * flatCoord_x + flatCoord_y * flatCoord_y);

		coords[index_x] = distance;
		coords[index_y] = flatCoord_y;
	}
}


__device__ float4 readValue(cudaSurfaceObject_t surfaceObject, int x, int y)
{
	float4 value;
	surf2Dread(&value, surfaceObject, sizeof(float4) * x, y);
	return value;
}


__device__ void writeValue(cudaSurfaceObject_t surfaceObject, float4 value, int x, int y)
{
	surf2Dwrite(value, surfaceObject, sizeof(float4) * x, y);
}


__global__ void renderTextureKernel(cudaSurfaceObject_t surfaceObject, double* coords, int textureSize_x, int textureSize_y)
{
	int textureCoord_x = blockIdx.x * blockDim.x + threadIdx.x;
	int textureCoord_y = blockIdx.y * blockDim.y + threadIdx.y;

	int index_x = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y, 0);
	int index_y = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y, 1);

	if (index_x != -1 && index_y != -1) // make sure the indices are valid
	{
		double coord_x = coords[index_x];
		double coord_y = coords[index_y];

		bool isOnLine = false;

		int index_right_x = getIndex(textureSize_x, textureSize_y, textureCoord_x + 1, textureCoord_y, 0);
		int index_right_y = getIndex(textureSize_x, textureSize_y, textureCoord_x + 1, textureCoord_y, 1);
		int index_up_x = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y + 1, 0);
		int index_up_y = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y + 1, 1);
		if (index_right_x != -1)
		{
			isOnLine = isOnLine || floor(coord_x) != floor(coords[index_right_x]) || floor(coord_y) != floor(coords[index_right_y]);
		}
		if (index_up_x != -1)
		{
			isOnLine = isOnLine || floor(coord_x) != floor(coords[index_up_x]) || floor(coord_y) != floor(coords[index_up_y]);
		}

		float line = (float)isOnLine;

		float4 value = make_float4(line, line, line, 1.0);
		writeValue(surfaceObject, value, textureCoord_x, textureCoord_y);
	}
}