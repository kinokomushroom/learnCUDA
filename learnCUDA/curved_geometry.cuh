#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>


__device__ const double PI = 3.14159265358979323846;
__device__ const double PIXEL_DISTANCE_STEP = 1.0; // proportional to pixel
__device__ const double DISTANCE_STEP_PRECISION = 1.0; // makes distance step more precise near extreme coordinates
__device__ const double DERIVATIVE_STEP = 0.01; // used when taking derivative of metric tensor

__device__ enum DisplayMode {
	LINES,
	COLOR_LINES
};


// ---------------- metrics ----------------
__device__ __host__ void sphereMetric(double* metric, double x, double y)
{
	double radius = 1.0;
	double halfCircumference = 4.0;

	// xx component
	metric[0] = radius * radius;
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	double sin_x = sin(x * PI / halfCircumference);
	metric[3] = radius * radius * sin_x * sin_x;
}

__device__ __host__ void torusMetric(double* metric, double x, double y) // https://www.cefns.nau.edu/~schulz/torus.pdf x and y are directions around the large and small radius
{
	double largeRadius = 3.0;
	double smallRadius = 1.0;
	double halfCircumference = 2.0;

	// xx component
	double cos_y = cos(y * PI / halfCircumference);
	metric[0] = (largeRadius + smallRadius * cos_y) * (largeRadius + smallRadius * cos_y);
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = smallRadius * smallRadius;
}

__device__ __host__ void poincareMetric(double* metric, double x, double y) // https://math.stackexchange.com/questions/1292707/comparing-metric-tensors-of-the-poincare-and-the-klein-disk-models-of-hyperbolic
{
	double divisor = (1.0 - x * x - y * y);
	double scale = 4.0;
	// xx component
	metric[0] = scale / divisor;
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = scale / divisor;
}

__device__ __host__ void kleinMetric(double* metric, double x, double y)
{
	double divisor = (1.0 - x * x - y * y);
	double scale = 16.0;
	// xx component
	metric[0] = scale / divisor + (x * x) / (divisor * divisor);
	// xy component
	metric[1] = (x * y) / (divisor * divisor);
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = scale / divisor + (y * y) / (divisor * divisor);
}

__device__ __host__ void euclideanMetric(double* metric, double x, double y)
{
	// xx component
	metric[0] = 1.0;
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = 1.0;
}

__device__ __host__ void minkowskiMetric(double* metric, double x, double y) 
{
	// xx component
	metric[0] = -1.0;
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = 1.0;
}

__device__ __host__ void schwarzschildMetric(double* metric, double x, double y)
{
	double schwarzschildRadius = 0.5;

	// xx component (space)
	metric[0] = -1.0 / (1.0 - schwarzschildRadius / x);
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component (time)
	metric[3] = (1.0 - schwarzschildRadius / x);
}
// ---------------- metrics ----------------



void printArray(double* doubleArray, int length, std::string name)
{
	std::cout << name << std::endl;
	for (int index = 0; index < length; index++)
	{
		std::cout << index << ": " << doubleArray[index] << std::endl;
	}
}


__device__ int getIndex(int textureSize_x, int textureSize_y, int textureCoord_x, int textureCoord_y, int axis)
{
	int index = 2 * (textureSize_x * textureCoord_y + textureCoord_x) + axis;
	bool isValid = textureCoord_x >= 0 && textureCoord_x < textureSize_x && textureCoord_y >= 0 && textureCoord_y < textureSize_y;
	index = isValid * index + (1 - isValid) * -1;
	return index;
}


__device__ __host__ double dotProduct(double* vector_a, double* vector_b, double* metric)
{
	double result = 0.0;
	for (int a = 0; a < 2; a++)
	{
		for (int b = 0; b < 2; b++)
		{
			result += vector_a[a] * vector_b[b] * metric[2 * a + b];
		}
	}
	return result;
}


__device__ __host__ void calculateMetric(double* metric, double x, double y) // metric[a, b] = metric[2 * a + b]
{
	sphereMetric(metric, x, y);
	//torusMetric(metric, x, y);
	//poincareMetric(metric, x, y);
	//kleinMetric(metric, x, y);
	//euclideanMetric(metric, x, y);
	//minkowskiMetric(metric, x, y);
	//schwarzschildMetric(metric, x, y);
}


__device__ __host__ void calculateMetricInverse(double* metricInverse, double* metric) // metricInverse[a, b] = metricInverse[2 * a + b]
{
	double determinant = metric[0] * metric[3] - metric[1] * metric[2];
	// xx component
	metricInverse[0] = metric[3] / determinant;
	// xy component
	metricInverse[1] = -metric[1] / determinant;
	// yx component
	metricInverse[2] = metricInverse[1];
	// yy component
	metricInverse[3] = metric[0] / determinant;
}


__device__ __host__ void calculateMetricJacobian(double* metricJacobian, double* position) // metricJacibian[a, b, k] = metricJacobian[4 * k + 2 * a + b]
{
	double metric_x_plus[4];
	double metric_x_minus[4];
	double metric_y_plus[4];
	double metric_y_minus[4];
	calculateMetric(metric_x_plus, position[0] + DERIVATIVE_STEP, position[1]);
	calculateMetric(metric_x_minus, position[0] - DERIVATIVE_STEP, position[1]);
	calculateMetric(metric_y_plus, position[0], position[1] + DERIVATIVE_STEP);
	calculateMetric(metric_y_minus, position[0], position[1] - DERIVATIVE_STEP);

	double doubleStep = 2 * DERIVATIVE_STEP;
	// d(g_xx) / dx
	metricJacobian[0] = (metric_x_plus[0] - metric_x_minus[0]) / doubleStep;
	// d(g_xx) / dy
	metricJacobian[4] = (metric_y_plus[0] - metric_y_minus[0]) / doubleStep;
	// d(g_xy) / dx
	metricJacobian[1] = (metric_x_plus[1] - metric_x_minus[1]) / doubleStep;
	// d(g_xy) / dy
	metricJacobian[5] = (metric_y_plus[1] - metric_y_minus[1]) / doubleStep;
	// d(g_yx) / dx
	metricJacobian[2] = metricJacobian[1];
	// d(g_yx) / dy
	metricJacobian[6] = metricJacobian[5];
	// d(g_yy) / dx
	metricJacobian[3] = (metric_x_plus[3] - metric_x_minus[3]) / doubleStep;
	// d(g_yy) / dy
	metricJacobian[7] = (metric_y_plus[3] - metric_y_minus[3]) / doubleStep;
}


__device__ __host__ void calculateChristoffel(double* christoffel, double* position) // christoffel[i, a, b] = christoffel[4 * i + 2 * a + b]
{
	double metric[4];
	calculateMetric(metric, position[0], position[1]);
	double metricInverse[4];
	calculateMetricInverse(metricInverse, metric);
	double metricJacobian[8];
	calculateMetricJacobian(metricJacobian, position);

	// christoffel[i, a, b] = 0.5 * metricInverse[j, i] * (metricJacobian[j, a, b] + metricJacobian[b, j, a] - metricJacobian[a, b, j])
	for (int i = 0; i < 2; i++)
	{
		for (int a = 0; a < 2; a++)
		{
			for (int b = 0; b < 2; b++)
			{
				double sum = 0;
				for (int j = 0; j < 2; j++)
				{
					sum += metricInverse[2 * j + i] * (metricJacobian[4 * b + 2 * j + a] + metricJacobian[4 * a + 2 * b + j] - metricJacobian[4 * j + 2 * a + b]);
				}
				christoffel[4 * i + 2 * a + b] = 0.5 * sum;
			}
		}
	}
}


__device__ __host__ void parallelTransport(double* targetVector, double* movementVector, double* position)
{
	double christoffel[8];
	calculateChristoffel(christoffel, position);

	double updatedTargetVector[2] = { targetVector[0], targetVector[1]};
	for (int i = 0; i < 2; i++)
	{
		for (int a = 0; a < 2; a++)
		{
			for (int b = 0; b < 2; b++)
			{
				updatedTargetVector[i] += -christoffel[4 * i + 2 * a + b] * targetVector[a] * movementVector[b];
			}
		}
	}

	targetVector[0] = updatedTargetVector[0];
	targetVector[1] = updatedTargetVector[1];
}


__device__ __host__ void updateVelocity(double* velocity, double* position, double differentiationStep, double& deltaTime, double baseDeltaTime)
{
	// parallel transport velocity by deltaTime * velocity
	double movementVector[2] = { deltaTime * velocity[0], deltaTime * velocity[1] };
	double originalVelocity[2] = { velocity[0], velocity[1] };
	parallelTransport(velocity, movementVector, position);

	// update deltaTime
	double acceleration[2] = { (velocity[0] - originalVelocity[0]) / deltaTime, (velocity[1] - originalVelocity[1]) / deltaTime };
	double accelerationLog = log(acceleration[0] * acceleration[0] + acceleration[1] * acceleration[1] + 1);
	deltaTime = baseDeltaTime / (1.0 + DISTANCE_STEP_PRECISION * accelerationLog);
}


__global__ void calculateCoords(double* coords, int textureSize_x, int textureSize_y, double scale_x, double scale_y, double basis_xx, double basis_yx, double basis_xy, double basis_yy, double position_x, double position_y)
{
	int textureCoord_x = blockIdx.x * blockDim.x + threadIdx.x;
	int textureCoord_y = blockIdx.y * blockDim.y + threadIdx.y;

	int index_x = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y, 0);
	int index_y = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y, 1);

	if (index_x != -1 && index_y != -1) // make sure the indices are valid
	{
		double textureCoordCentered_x = textureCoord_x - textureSize_x / 2;
		double textureCoordCentered_y = textureCoord_y - textureSize_y / 2;

		double flatCoord_x = ((double)textureCoord_x / textureSize_x * 2.0 - 1.0) * scale_x;
		double flatCoord_y = ((double)textureCoord_y / textureSize_y * 2.0 - 1.0) * scale_y;

		double textureDistance = sqrt(textureCoordCentered_x * textureCoordCentered_x + textureCoordCentered_y * textureCoordCentered_y);
		double flatDistance = sqrt(flatCoord_x * flatCoord_x + flatCoord_y * flatCoord_y);

		double flatDirection[2];
		flatDirection[0] = flatCoord_x / flatDistance;
		flatDirection[1] = flatCoord_y / flatDistance;

		double baseDeltaDistance = flatDistance / textureDistance * PIXEL_DISTANCE_STEP; // also d_tau
		double deltaDistance = baseDeltaDistance;
		double velocity[2];
		velocity[0] = flatDirection[0] * basis_xx + flatDirection[1] * basis_yx;
		velocity[1] = flatDirection[0] * basis_xy + flatDirection[1] * basis_yy;

		double curvedCoord[2];
		curvedCoord[0] = position_x;
		curvedCoord[1] = position_y;
		double travelledDistance = 0.0;

		double oldCurvedCoord[2];
		while (travelledDistance < flatDistance - deltaDistance)
		{
			// record previous position
			oldCurvedCoord[0] = curvedCoord[0];
			oldCurvedCoord[1] = curvedCoord[1];

			// update position
			curvedCoord[0] += deltaDistance * velocity[0];
			curvedCoord[1] += deltaDistance * velocity[1];
			travelledDistance += deltaDistance;

			updateVelocity(velocity, oldCurvedCoord, 0.01, deltaDistance, baseDeltaDistance);
		}
		double remainingDistance = flatDistance - (travelledDistance);
		curvedCoord[0] += remainingDistance * velocity[0];
		curvedCoord[1] += remainingDistance * velocity[1];

		coords[index_x] = curvedCoord[0];
		coords[index_y] = curvedCoord[1];
	}
}


// initialize basis with the x basis vector aligned with coordinate x direction
__host__ void initializeBasis(double* position, double* basis)
{
	double metric[4];
	calculateMetric(metric, position[0], position[1]);
	double coordBasis_xLength = sqrt(abs(metric[0]));
	double coordBasis_yLength = sqrt(abs(metric[3]));
	double coordBasisAngle = acos(metric[1] / (coordBasis_xLength * coordBasis_yLength));

	double forwardTransform[4]; // forwardTransformation[a, b] = forwardTransformation[2 * a + b], transformation from orthonormal basis to coordinate basis
	forwardTransform[0] = coordBasis_xLength;
	forwardTransform[2] = 0.0;
	forwardTransform[1] = coordBasis_yLength * cos(coordBasisAngle);
	forwardTransform[3] = coordBasis_yLength * sin(coordBasisAngle);

	double determinant = forwardTransform[0] * forwardTransform[3] - forwardTransform[1] * forwardTransform[2];
	basis[0] = forwardTransform[3] / determinant;
	basis[1] = -forwardTransform[1] / determinant;
	basis[2] = -forwardTransform[2] / determinant;
	basis[3] = forwardTransform[0] / determinant;
}


// fix slight errors from parallel transport and make basis orthonormal
__host__ void fixBasis(double* position, double* basis)
{
	double metric[4];
	calculateMetric(metric, position[0], position[1]);

	double basis_x[2] = { basis[0], basis[2] };
	double basis_y[2] = { basis[1], basis[3] };
	double basis_xLength = sqrt(abs(dotProduct(basis_x, basis_x, metric)));
	// normalize x basis
	basis_x[0] /= basis_xLength;
	basis_x[1] /= basis_xLength;

	// fix y basis in the x direction so they become orthogonal
	double basis_yFix = dotProduct(basis_x, basis_y, metric);
	basis_y[0] += -basis_yFix * basis_x[0];
	basis_y[1] += -basis_yFix * basis_x[1];

	// normalize y basis
	double basis_yLength = sqrt(abs(dotProduct(basis_y, basis_y, metric)));
	basis_y[0] /= basis_yLength;
	basis_y[1] /= basis_yLength;

	// debug
	//double basis_xyDot = dotProduct(basis_x, basis_y, metric);
	//std::cout << "basis x y dot: " << basis_xyDot << std::endl;

	basis[0] = basis_x[0];
	basis[2] = basis_x[1];
	basis[1] = basis_y[0];
	basis[3] = basis_y[1];
}


__host__ void updatePosition(double* position, double* basis, int* input, double moveDistance)
{
	// create movement vector from input 
	double inputMagnitude = sqrt(input[0] * input[0] + input[1] * input[1]);
	double movementVectorFlat[2] = { input[0], input[1] };
	if (inputMagnitude != 0)
	{
		movementVectorFlat[0] /= inputMagnitude;
		movementVectorFlat[1] /= inputMagnitude;
	}
	movementVectorFlat[0] *= moveDistance;
	movementVectorFlat[1] *= moveDistance;

	// convert movement vector to current basis
	double movementVector[2] = { movementVectorFlat[0] * basis[0] + movementVectorFlat[1] * basis[1], movementVectorFlat[0] * basis[2] + movementVectorFlat[1] * basis[3] };

	// update position
	double oldPositon[2] = { position[0], position[1] };
	position[0] += movementVector[0];
	position[1] += movementVector[1];

	// parallel transport basis vectors
	double basis_x[2] = { basis[0], basis[2] };
	double basis_y[2] = { basis[1], basis[3] };
	parallelTransport(basis_x, movementVector, oldPositon);
	parallelTransport(basis_y, movementVector, oldPositon);
	basis[0] = basis_x[0];
	basis[2] = basis_x[1];
	basis[1] = basis_y[0];
	basis[3] = basis_y[1];

	fixBasis(position, basis);
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


__global__ void renderTextureKernel(cudaSurfaceObject_t surfaceObject, double* coords, int textureSize_x, int textureSize_y, int displayMode)
{
	int textureCoord_x = blockIdx.x * blockDim.x + threadIdx.x;
	int textureCoord_y = blockIdx.y * blockDim.y + threadIdx.y;
	float normalziedCoord_x = (float)textureCoord_x / textureSize_x * 2.0 - 1.0;
	float normalziedCoord_y = (float)textureCoord_y / textureSize_y * 2.0 - 1.0;

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

		float4 value = make_float4(0, 0, 0, 0);
		if (displayMode == LINES)
		{
			value = make_float4(line, line, line, 1.0); // just lines
		}
		else if (displayMode == COLOR_LINES)
		{
			value = make_float4(fmax((float)coord_x, line), fmax((float)coord_y, line), line, 1.0); // coordinates with lines
		}

		if (sqrt(normalziedCoord_x * normalziedCoord_x + normalziedCoord_y * normalziedCoord_y) < 0.03) // draw center position
		{
			value = make_float4(0.0, 1.0, 1.0, 1.0);
		}

		writeValue(surfaceObject, value, textureCoord_x, textureCoord_y);
	}
}