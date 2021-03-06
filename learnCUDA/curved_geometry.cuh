#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>


__device__ const double PI = 3.14159265358979323846;
__device__ const double DERIVATIVE_STEP = 0.001; // used when taking derivative of metric tensor

__device__ enum DisplayMode {
	LINES,
	COLOR_LINES,
	REPEATED_COLOR_LINES,
};


// ---------------- metrics ----------------
__device__ __host__ void sphereMetric(double* metric, double x, double y)
{
	double radius = 1.0;

	// xx component
	metric[0] = radius * radius;
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	double sin_x = sin(x);
	metric[3] = radius * radius * sin_x * sin_x;
}

__device__ __host__ void torusMetric(double* metric, double x, double y) // https://www.cefns.nau.edu/~schulz/torus.pdf x and y are directions around the large and small radius
{
	double largeRadius = 3.0;
	double smallRadius = 1.0;

	// xx component
	double cos_y = cos(y);
	metric[0] = (largeRadius + smallRadius * cos_y) * (largeRadius + smallRadius * cos_y);
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = smallRadius * smallRadius;
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

__device__ __host__ void hyperboloidMetric(double* metric, double x, double y) // https://en.wikipedia.org/wiki/Poincar%C3%A9_metric
{
	double divisor = y * y;
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

__device__ __host__ void poincareMetric(double* metric, double x, double y) // https://math.stackexchange.com/questions/1292707/comparing-metric-tensors-of-the-poincare-and-the-klein-disk-models-of-hyperbolic
{
	double divisor = 1.0 - x * x - y * y;
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

__device__ __host__ void schwarzschildMetric(double* metric, double x, double y) // https://en.wikipedia.org/wiki/Metric_tensor
{
	// (x, y) = (r, t)
	double schwarzschildRadius = 4.0; // x = schwarzschildRadius is event horizon, and x = 0 is singularity

	// xx component (space)
	metric[0] = -1.0 / (1.0 - schwarzschildRadius / x);
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component (time)
	metric[3] = (1.0 - schwarzschildRadius / x);
}

__device__ __host__ void schwarzschildSpatialMetric(double* metric, double x, double y)
{
	// (x, y) = (r, theta)
	double schwarzschildRadius = 2.0; // x = 0 is event horizon, and x = -schwarzSchildRadius is singularity

	// xx component (space)
	metric[0] = -1.0 / (1.0 - schwarzschildRadius / x);
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component (time)
	metric[3] = -x * x;
}

__device__ __host__ void wormholeMetric(double* metric, double x, double y) // https://en.wikipedia.org/wiki/Ellis_wormhole
{
	double radius = 1.0;
	// xx component
	metric[0] = 1.0;
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = x * x + radius * radius;
}

__device__ __host__ void kerrMetric(double* metric, double x, double y) // https://en.wikipedia.org/wiki/Kerr_metric
{
	// (x, y) = (phi, t) , theta = 2 / pi
	double r = 3.0;
	double schwarzschildRadius = 2.0;
	double angularDivMass = 0.8;
	double sigma = r * r;
	//double delta = x * x - schwarzschildRadius * x + angularDivMass * angularDivMass;
	// xx component
	metric[0] = r * r + angularDivMass * angularDivMass * (1 + schwarzschildRadius * r / sigma);
	// xy component
	metric[1] = -2 * schwarzschildRadius * r * angularDivMass / sigma;
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = -1 + schwarzschildRadius * r / sigma;
}

__device__ __host__ void kerrSpatialMetric(double* metric, double x, double y)
{
	// (x, y) = (r, phi) , theta = pi / 2
	double schwarzschildRadius = 2.0;
	double angularDivMass = 0.5;
	double sigma = x * x;
	double delta = x * x - schwarzschildRadius * x + angularDivMass * angularDivMass;
	// xx component
	metric[0] = sigma / delta;
	// xy component
	metric[1] = 0.0;
	// yx component
	metric[2] = metric[1];
	// yy component
	metric[3] = (x * x + angularDivMass * angularDivMass * (1 + schwarzschildRadius * x));
}
// ---------------- metrics ----------------

__device__ enum CoordMode
{
	NONE,
	CYCLIC,
	MIRROR,
};
__device__ __host__ double convertCoords(double value, int mode, double scale, double cycle)
{
	double newValue = value * scale;
	if (mode == NONE)
	{
		return newValue;
	}
	if (mode == CYCLIC)
	{
		newValue = newValue - floor(newValue / cycle) * cycle;
		if (newValue > cycle / 2)
		{
			newValue -= cycle;
		}
		return newValue;
	}
	if (mode == MIRROR)
	{
		double doubleCycle = 2 * cycle;
		newValue = newValue - floor(newValue / doubleCycle) * doubleCycle;
		if (newValue > cycle)
		{
			newValue = -newValue + 2 * cycle;
		}
		return newValue;
	}
}

typedef void(*metricFunction_t)(double* metric, double x, double y);
const int METRIC_FUNCTION_COUNT = 11;
__device__ const metricFunction_t metricFunctions[] =
{
	euclideanMetric,
	minkowskiMetric,
	sphereMetric,
	torusMetric,
	hyperboloidMetric,
	poincareMetric,
	schwarzschildMetric,
	schwarzschildSpatialMetric,
	wormholeMetric,
	kerrMetric,
	kerrSpatialMetric,
};

struct MetricInfo
{
	std::string name;
	std::string description;
	double initialPosition[2];
	bool isCoordCyclic[2];
	int coordMode[2];
	double coordScale[2];
	double coordCycle[2];
	MetricInfo(char* _name, std::string gxx, std::string gxy, std::string gyy, double initialPosition_0, double initialPosition_1, int coordMode_0, int coordMode_1, double coordScale_0, double coordScale_1, double coordCycle_0, double coordCycle_1)
	{
		name = _name;
		description = "g_xx = " + gxx + "\ng_xy = " + gxy + "\ng_yy = " + gyy;
		initialPosition[0] = initialPosition_0;
		initialPosition[1] = initialPosition_1;
		coordMode[0] = coordMode_0;
		coordMode[1] = coordMode_1;
		coordScale[0] = coordScale_0;
		coordScale[1] = coordScale_1;
		coordCycle[0] = coordCycle_0;
		coordCycle[1] = coordCycle_1;
	}
};
const MetricInfo metricInfos[] =
{
	MetricInfo("Euclidean", "1", "0", "1", 0, 0, NONE, NONE, 1, 1, 0, 0),
	MetricInfo("Minkowski", "-1", "0", "1", 0, 0, NONE, NONE, 1, 1, 0, 0),
	MetricInfo("Sphere", "r^2", "0", "(r * sin(x))^2", PI / 2, 0, MIRROR, CYCLIC, 4 / PI, 4 / PI, 4, 8),
	MetricInfo("Torus", "(R + r * cos(y))^2", "0", "r^2", 0, 0, CYCLIC, CYCLIC, 4 / PI, 2 / PI, 8, 4),
	MetricInfo("Hyperboloid", "r / y^2", "0", "r / y^2", 0, 1, NONE, NONE, 1, 1, 0, 0),
	MetricInfo("Poincare Disk", "r / (1 - x^2 - y^2)", "0", "r / (1 - x^2 - y^2)", 1, 1, NONE, NONE, 1, 1, 0, 0),
	MetricInfo("Schwarzschild", "-1 / (1 - R / x)", "0", "1 - R / x", 6, 0, NONE, NONE, 1, 1, 0, 0),
	MetricInfo("Schwarzschild (Spatial)", "-1 / (1 - R / x)", "0", "-x^2", 3, 0, NONE, CYCLIC, 1, 4 / PI, 0, 8),
	MetricInfo("Wormhole", "1", "0", "x^2 + r^2", 2, 0, NONE, CYCLIC, 1, 4 / PI, 0, 8),
	MetricInfo("Kerr", "r^2 + a^2 * (1 + R / r)", "2 * a * R / r", "1 - R / r", 0, 0, CYCLIC, NONE, 4 / PI, 1, 8, 0),
	MetricInfo("Kerr (Spatial)", "x^2 / (x^2 - R * x + a^2)", "0", "x^2 + a^2 * (1 + R / x)", 3, 0, NONE, CYCLIC, 1, 4 / PI, 0, 8),
};


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

__device__ __host__ void calculateMetric(double* metric, double x, double y, int metricFunctionIndex) // metric[a, b] = metric[2 * a + b]
{
	(*(metricFunctions[metricFunctionIndex]))(metric, x, y);
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


__device__ __host__ void calculateMetricJacobian(double* metricJacobian, double* position, int metricFunctionIndex) // metricJacibian[a, b, k] = metricJacobian[4 * k + 2 * a + b]
{
	double metric_x_plus[4];
	double metric_x_minus[4];
	double metric_y_plus[4];
	double metric_y_minus[4];
	calculateMetric(metric_x_plus, position[0] + DERIVATIVE_STEP, position[1], metricFunctionIndex);
	calculateMetric(metric_x_minus, position[0] - DERIVATIVE_STEP, position[1], metricFunctionIndex);
	calculateMetric(metric_y_plus, position[0], position[1] + DERIVATIVE_STEP, metricFunctionIndex);
	calculateMetric(metric_y_minus, position[0], position[1] - DERIVATIVE_STEP, metricFunctionIndex);

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


__device__ __host__ void calculateChristoffel(double* christoffel, double* position, int metricFunctionIndex) // christoffel[i, a, b] = christoffel[4 * i + 2 * a + b]
{
	double metric[4];
	calculateMetric(metric, position[0], position[1], metricFunctionIndex);
	double metricInverse[4];
	calculateMetricInverse(metricInverse, metric);
	double metricJacobian[8];
	calculateMetricJacobian(metricJacobian, position, metricFunctionIndex);

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


__device__ __host__ void parallelTransport(double* targetVector, double* movementVector, double* position, int metricFunctionIndex)
{
	double christoffel[8];
	calculateChristoffel(christoffel, position, metricFunctionIndex);

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


__device__ __host__ void updateVelocity(double* velocity, double* position, double differentiationStep, double& deltaTime, double baseDeltaTime, double distanceStepPrecision, double maxPrecision, int metricFunctionIndex)
{
	// parallel transport velocity by deltaTime * velocity
	double movementVector[2] = { deltaTime * velocity[0], deltaTime * velocity[1] };
	double originalVelocity[2] = { velocity[0], velocity[1] };
	parallelTransport(velocity, movementVector, position, metricFunctionIndex);

	// update deltaTime
	double acceleration[2] = { (velocity[0] - originalVelocity[0]) / deltaTime, (velocity[1] - originalVelocity[1]) / deltaTime };
	double accelerationLog = min(log(acceleration[0] * acceleration[0] + acceleration[1] * acceleration[1] + 1), maxPrecision);
	deltaTime = baseDeltaTime / (1.0 + distanceStepPrecision * accelerationLog);
}


__global__ void calculateCoords(double* coords, int textureSize_x, int textureSize_y, double scale_x, double scale_y, double basis_xx, double basis_yx, double basis_xy, double basis_yy, double position_x, double position_y, double pixelDistanceStep, double distanceStepPrecision, double maxPrecision, int metricFunctionIndex)
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

		double baseDeltaDistance = flatDistance / textureDistance * pixelDistanceStep; // also d_tau
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

			updateVelocity(velocity, oldCurvedCoord, 0.01, deltaDistance, baseDeltaDistance, distanceStepPrecision, maxPrecision, metricFunctionIndex);
		}
		double remainingDistance = flatDistance - (travelledDistance);
		curvedCoord[0] += remainingDistance * velocity[0];
		curvedCoord[1] += remainingDistance * velocity[1];

		coords[index_x] = curvedCoord[0];
		coords[index_y] = curvedCoord[1];
	}
}


// fix slight errors from parallel transport and make basis orthonormal
__host__ void fixBasis(double* position, double* basis, int metricFunctionIndex)
{
	double metric[4];
	calculateMetric(metric, position[0], position[1], metricFunctionIndex);

	double basis_x[2] = { basis[0], basis[2] };
	double basis_y[2] = { basis[1], basis[3] };
	double basis_xLength = sqrt(abs(dotProduct(basis_x, basis_x, metric)));
	// normalize x basis
	basis_x[0] /= basis_xLength;
	basis_x[1] /= basis_xLength;

	// fix y basis in the x direction so they become orthogonal
	double basis_yFix = dotProduct(basis_x, basis_y, metric);
	double testBasis_y[2] = { basis_y[0] - basis_yFix * basis_x[0], basis_y[1] - basis_yFix * basis_x[1] };
	double basis_yFixResult = dotProduct(basis_x, testBasis_y, metric);
	if (abs(basis_yFixResult) < abs(basis_yFix)) // check if fix direction was valid
	{
		basis_y[0] += -basis_yFix * basis_x[0];
		basis_y[1] += -basis_yFix * basis_x[1];
	}
	else
	{
		basis_y[0] += basis_yFix * basis_x[0];
		basis_y[1] += basis_yFix * basis_x[1];
	}

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


// initialize basis with the x basis vector aligned with coordinate x direction
__host__ void initializeBasis(double* position, double* basis, int metricFunctionIndex)
{
	basis[0] = 1;
	basis[1] = 0;
	basis[2] = 0;
	basis[3] = 1;

	fixBasis(position, basis, metricFunctionIndex);
}


__host__ void updatePosition(double* position, double* basis, int* input, double moveDistance, int inputRotation, double rotationAngle, int metricFunctionIndex)
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
	parallelTransport(basis_x, movementVector, oldPositon, metricFunctionIndex);
	parallelTransport(basis_y, movementVector, oldPositon, metricFunctionIndex);
	basis[0] = basis_x[0];
	basis[2] = basis_x[1];
	basis[1] = basis_y[0];
	basis[3] = basis_y[1];

	// rotate basis
	double metric[4];
	calculateMetric(metric, position[0], position[1], metricFunctionIndex);
	double localMetric[4] = { dotProduct(basis_x, basis_x, metric), dotProduct(basis_x, basis_y, metric), dotProduct(basis_y, basis_x, metric), dotProduct(basis_y, basis_y, metric) };
	//printArray(localMetric, 4, "local metric");
	double rotationComponents[4]; // rotationComponents[a, b] = rotationComponents[2 * a + b]
	rotationComponents[2] = 1; // PHI[1, 0] = 1
	rotationComponents[1] = (localMetric[3] * (localMetric[1] - localMetric[0] * localMetric[3])) / (localMetric[0] * (localMetric[0] * localMetric[3] - localMetric[2])) * rotationComponents[2]; // PHI[0, 1] = (g[1, 1] * (g[0, 1] - g[0, 0] * g[1, 1])) / (g[0, 0] * (g[0, 0] * g[1, 1] - g[1, 0])) * PHI[1, 0]
	rotationComponents[0] = -localMetric[1] / localMetric[0] * rotationComponents[2]; // PHI[0, 0] = -g[0, 1] / g[0, 0] * PHI[1, 0]
	rotationComponents[3] = -localMetric[2] / localMetric[3] * rotationComponents[1]; // PHI[1, 1] = -g[1, 0] / g[1, 1] * PHI[0, 1]
	//printArray(rotationComponents, 4, "rot components");
	double rotationVector_x[2] = { rotationComponents[0] * basis[0] + rotationComponents[2] * basis[1], rotationComponents[0] * basis[2] + rotationComponents[2] * basis[3] };
	double rotationVector_y[2] = { rotationComponents[1] * basis[0] + rotationComponents[3] * basis[1], rotationComponents[1] * basis[2] + rotationComponents[3] * basis[3] };
	//printArray(rotationVector_x, 2, "rot x");
	//printArray(rotationVector_y, 2, "rot y");
	basis[0] += inputRotation * rotationAngle * rotationVector_x[0];
	basis[2] += inputRotation * rotationAngle * rotationVector_x[1];
	basis[1] += inputRotation * rotationAngle * rotationVector_y[0];
	basis[3] += inputRotation * rotationAngle * rotationVector_y[1];

	fixBasis(position, basis, metricFunctionIndex);
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


__global__ void renderTextureKernel(cudaSurfaceObject_t surfaceObject, double* coords, int textureSize_x, int textureSize_y, int displayMode, double position_0, double position_1, double initialPosition_0, double initialPosition_1, int metricFunctionIndex, double pointSize, double scale, int coordMode_0, int coordMode_1, double coordScale_0, double coordScale_1, double coordCycle_0, double coordCycle_1)
{
	int textureCoord_x = blockIdx.x * blockDim.x + threadIdx.x;
	int textureCoord_y = blockIdx.y * blockDim.y + threadIdx.y;
	float normalziedCoord_x = (float)textureCoord_x / textureSize_x * 2.0 - 1.0;
	float normalziedCoord_y = (float)textureCoord_y / textureSize_y * 2.0 - 1.0;

	int index_x = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y, 0);
	int index_y = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y, 1);

	if (index_x != -1 && index_y != -1) // make sure the indices are valid
	{
		double coord_x = convertCoords(coords[index_x], coordMode_0, coordScale_0, coordCycle_0);
		double coord_y = convertCoords(coords[index_y], coordMode_1, coordScale_1, coordCycle_1);

		bool isOnLine = false;

		int index_right_x = getIndex(textureSize_x, textureSize_y, textureCoord_x + 1, textureCoord_y, 0);
		int index_right_y = getIndex(textureSize_x, textureSize_y, textureCoord_x + 1, textureCoord_y, 1);
		int index_up_x = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y + 1, 0);
		int index_up_y = getIndex(textureSize_x, textureSize_y, textureCoord_x, textureCoord_y + 1, 1);
		double coord_right_x = convertCoords(coords[index_right_x], coordMode_0, coordScale_0, coordCycle_0);
		double coord_right_y = convertCoords(coords[index_right_y], coordMode_1, coordScale_1, coordCycle_1);
		double coord_up_x = convertCoords(coords[index_up_x], coordMode_0, coordScale_0, coordCycle_0);
		double coord_up_y = convertCoords(coords[index_up_y], coordMode_1, coordScale_1, coordCycle_1);
		if (index_right_x != -1)
		{
			isOnLine = isOnLine || floor(coord_x) != floor(coord_right_x) || floor(coord_y) != floor(coord_right_y);
		}
		if (index_up_x != -1)
		{
			isOnLine = isOnLine || floor(coord_x) != floor(coord_up_x) || floor(coord_y) != floor(coord_up_y);
		}

		float line = (float)isOnLine;

		float4 value = make_float4(0, 0, 0, 0);
		if (displayMode == LINES)
		{
			value = make_float4(line, line, line, 1.0); // just lines
		}
		else if (displayMode == COLOR_LINES)
		{
			value = make_float4(max(coord_x, line), max(coord_y, line), line, 1.0); // coordinates with lines
		}
		else if (displayMode == REPEATED_COLOR_LINES)
		{
			value = make_float4(max(coord_x - floor(coord_x), line), max(coord_y - floor(coord_y), line), line, 1.0); // coordinates with lines
		}

		// render circle in initial viewer position
		double initialMetric[4];
		calculateMetric(initialMetric, initialPosition_0, initialPosition_1, metricFunctionIndex);
		double initialViewerDirection[2] = { convertCoords(coords[index_x] - initialPosition_0, coordMode_0, 1, coordCycle_0 / coordScale_0), convertCoords(coords[index_y] - initialPosition_1, coordMode_1, 1, coordCycle_1 / coordScale_1) };
		double initialViewerDistance = sqrt(abs(dotProduct(initialViewerDirection, initialViewerDirection, initialMetric)));
		if (initialViewerDistance <= pointSize * scale) // draw center position
		{
			value = make_float4(1.0, 0.0, 1.0, 1.0);
		}

		// render circle in viewer position
		double metric[4];
		calculateMetric(metric, position_0, position_1, metricFunctionIndex);
		double viewerDirection[2] = { convertCoords(coords[index_x] - position_0, coordMode_0, 1, coordCycle_0 / coordScale_0), convertCoords(coords[index_y] - position_1, coordMode_1, 1, coordCycle_1 / coordScale_1) };
		double viewerDistance = sqrt(abs(dotProduct(viewerDirection, viewerDirection, metric)));
		if (viewerDistance <= pointSize * scale) // draw center position
		{
			value = make_float4(0.0, 1.0, 1.0, 1.0);
		}

		writeValue(surfaceObject, value, textureCoord_x, textureCoord_y);
	}
}