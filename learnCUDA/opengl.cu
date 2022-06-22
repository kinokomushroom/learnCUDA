#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "stb_image.h"

#include "shader.h"
#include "curved_geometry.cuh"


GLFWwindow* initOpenGL();
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
unsigned int createTexture(int textureSize_x, int textureSize_y);
unsigned int createQuadVAO();
void renderTextureCUDA(cudaGraphicsResource_t textureResource, double* coords, int textureSize_x, int textureSize_y);


const unsigned int SCREEN_WIDTH = 512;
const unsigned int SCREEN_HEIGHT = 512;
const unsigned int TEXTURE_SIZE = 512;

double deltaTime = 0.0;
double lastTime = 0.0;


int main()
{
	// initialize GLFW and GLAD, and create a window
	GLFWwindow* window;
	window = initOpenGL();
	if (window == NULL)
	{
		return -1;
	}

	// set callbacks
	glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

	// create texture to display
	unsigned int texture = createTexture(TEXTURE_SIZE, TEXTURE_SIZE);

	// register texture with CUDA
	cudaGraphicsResource_t textureResource;
	cudaGraphicsGLRegisterImage(&textureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	
	// register texture to shader
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	// set up VAO and VBO
	unsigned int VAO = createQuadVAO();

	// set up shader
	Shader shader = Shader("vertex.glsl", "fragment.glsl");


	double* coords;
	size_t bytes = 2 * TEXTURE_SIZE * TEXTURE_SIZE * sizeof(double);
	cudaMalloc(&coords, bytes);


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

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// update coords
		unsigned int blockSize = 16;
		dim3 blockDimension(blockSize, blockSize);
		dim3 gridDimension((int)ceil((float)TEXTURE_SIZE / blockSize), (int)ceil((float)TEXTURE_SIZE / blockSize));
		calculateCoords<<<gridDimension, blockDimension>>>(coords, TEXTURE_SIZE, TEXTURE_SIZE, 5.0, 5.0);
		cudaDeviceSynchronize();

		// render texture with CUDA
		renderTextureCUDA(textureResource, coords, TEXTURE_SIZE, TEXTURE_SIZE);

		// draw texture
		shader.use();
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	cudaFree(coords);

	glfwTerminate();
	return 0;
}


GLFWwindow* initOpenGL()
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
		return NULL;
	}
	glfwMakeContextCurrent(window);

	// initialize GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return NULL;
	}
	return window;
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


unsigned int createTexture(int textureSize_x, int textureSize_y)
{
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, textureSize_x, textureSize_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	return texture;
}


unsigned int createQuadVAO()
{
	// quad that fills screen
	float vertexData[] =
	{
		// position         uv
		-1.0, -1.0,  0.0,   0.0, 0.0,
		 1.0, -1.0,  0.0,   1.0, 0.0,
		-1.0,  1.0,  0.0,   0.0, 1.0,
		 1.0,  1.0,  0.0,   1.0, 1.0
	};

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

	return VAO;
}


void renderTextureCUDA(cudaGraphicsResource_t textureResource, double* coords, int textureSize_x, int textureSize_y)
{
	cudaGraphicsMapResources(1, &textureResource, 0);
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
	dim3 blockDimension(blockSize, blockSize);
	dim3 gridDimension((int)ceil((float)textureSize_x / blockSize), (int)ceil((float)textureSize_y / blockSize));
	renderTextureKernel<<<gridDimension, blockDimension>>>(surfaceObject, coords, TEXTURE_SIZE, TEXTURE_SIZE);
	cudaDeviceSynchronize();
	// free
	cudaDestroySurfaceObject(surfaceObject);
	//cudaFreeArray(textureArray); // DO NOT FREE ARRAY!!
	cudaGraphicsUnmapResources(1, &textureResource, 0);
}