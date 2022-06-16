#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include "stb_image.h"

#include "shader.h"


void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);


const unsigned int SCREEN_WIDTH = 512;
const unsigned int SCREEN_HEIGHT = 512;

double deltaTime = 0.0;
double lastTime = 0.0;


__global__ void testKernel(cudaSurfaceObject_t surfaceObject, int width, int height, double time)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		float4 value = make_float4((float)x / width, (float)y / height, sin(time) * 0.5 + 0.5, 1.0f);
		surf2Dwrite(value, surfaceObject, sizeof(float4) * x, y);
	}
}


int main()
{
	// initialize GLFW
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // use OpenGL version 4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// create a window
	GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// initialize GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// set callbacks
	glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

	// create texture
	const unsigned int TEXTURE_SIZE = 512;
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TEXTURE_SIZE, TEXTURE_SIZE, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	// register texture with CUDA
	cudaGraphicsResource_t textureResource;
	cudaGraphicsGLRegisterImage(&textureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	
	// register texture to shader
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	// quad that fills screen
	float vertexData[] =
	{
		// position         uv
		-1.0, -1.0,  0.0,   0.0, 0.0,
		 1.0, -1.0,  0.0,   1.0, 0.0,
		-1.0,  1.0,  0.0,   0.0, 1.0,
		 1.0,  1.0,  0.0,   1.0, 1.0
	};

	// set up VAO and VBO
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);
	unsigned int VBO;
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);
	// set up vertex attribute pointers
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	// unbind
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// set up shader
	Shader shader = Shader("vertex.glsl", "fragment.glsl");


	// render loop
	while (!glfwWindowShouldClose(window))
	{
		// calculate delta time
		double currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;
		if (floor(currentTime - deltaTime) < floor(currentTime))
		{
			std::cout << (int)floor(currentTime) << ": " << deltaTime << std::endl;
		}

		processInput(window);

		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// CUDA
		cudaGraphicsMapResources(1, &textureResource, 0);
		{
			// get mapped array
			cudaArray_t textureArray;
			cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0);
			// specify surface
			struct cudaResourceDesc resourceDescription;
			memset(&resourceDescription, 0, sizeof(resourceDescription));
			resourceDescription.resType = cudaResourceTypeArray;
			// create surface object
			resourceDescription.res.array.array = textureArray;
			cudaSurfaceObject_t surfaceObject;
			cudaCreateSurfaceObject(&surfaceObject, &resourceDescription);
			// launch kernel
			unsigned int blockSize = 16;
			unsigned int gridSize = (int)ceil((float)TEXTURE_SIZE / blockSize);
			dim3 blockDimension(blockSize, blockSize);
			dim3 gridDimension(gridSize, gridSize);
			testKernel<<<gridDimension, blockDimension>>>(surfaceObject, TEXTURE_SIZE, TEXTURE_SIZE, glfwGetTime());
			// free
			cudaDestroySurfaceObject(surfaceObject);
			//cudaFreeArray(textureArray); // DO NOT FREE ARRAY!!
		}
		cudaGraphicsUnmapResources(1, &textureResource, 0);

		// draw
		{
			shader.use();
			glBindVertexArray(VAO);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}


void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}


void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}
}