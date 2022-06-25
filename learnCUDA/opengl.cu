#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "stb_image.h"

#include <iostream>
#include <cmath>

#include "shader.h"
#include "curved_geometry.cuh"

GLFWwindow* initOpenGL();
void initImGui(ImGuiIO& io, GLFWwindow* window);
void glfwErrorCallback(int error, const char* description);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void initializePositionBasis();
void changeDisplayMode();
void changeMetricType(bool increment = false);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void processInput(GLFWwindow* window);
unsigned int createTexture(int textureSize_x, int textureSize_y);
unsigned int createQuadVAO();
void renderTextureCUDA(cudaGraphicsResource_t textureResource, double* coords, int textureSize_x, int textureSize_y);


const unsigned int SCREEN_WIDTH = 512;
const unsigned int SCREEN_HEIGHT = 256;
const unsigned int TEXTURE_SIZE = 256;

double deltaTime = 0.0;
double lastTime = 0.0;
const double fpsUpdateInterval = 0.2;
double recentCurrentTime = 0.0;
double recentDeltaTime = 0.0;
double elapsedTime = 0.0;

bool updateFrame = true;

double MOVE_DISTANCE = 0.1;
double ROTATION_ANGLE = 0.05;
int input[2] = { 0, 0 };
int inputRotation = 0;
double initialPosition[2] = { 1.0, 1.0 };
double position[2];
double basis[4]; // basis[a, b] = basis[2 * a + b];
double rotation = 0;

int displayMode = LINES;

bool displayImGuiWindow = true;

int METRIC_FUNCTION_INDEX = 2;
double MAGNIFICATION = 4.0;
double PIXEL_DISTANCE_STEP = 1.0; // proportional to pixel
double DISTANCE_STEP_PRECISION = 1.0; // makes distance step more precise near extreme coordinates
double MAX_PRECISION = 5.0;


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
	glfwSetKeyCallback(window, keyCallback);

	// set up Dear ImGui
	ImGuiIO io;
	initImGui(io, window);

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

	// allocate memory for texture to write coordinate data to
	double* coords;
	size_t bytes = 2 * TEXTURE_SIZE * TEXTURE_SIZE * sizeof(double);
	cudaMalloc(&coords, bytes);

	// initialize position and basis
	initializePositionBasis();

	// render loop
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		// calculate delta time
		double currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;
		if (currentTime - recentCurrentTime >= fpsUpdateInterval)
		{
			recentCurrentTime = currentTime;
			recentDeltaTime = deltaTime;
		}

		processInput(window);

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// start ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// show ImGui window
		if (displayImGuiWindow)
		{
			// variables
			float SPEED_ImGui = (float)MOVE_DISTANCE;
			float ANGULAR_ImGui = (float)ROTATION_ANGLE;
			float SCALE_ImGui = (float)MAGNIFICATION;
			float STEP_ImGui = (float)PIXEL_DISTANCE_STEP;
			float PRECISION_ImGui = (float)DISTANCE_STEP_PRECISION;
			float MAX_PRECISION_ImGui = (float)MAX_PRECISION;

			ImGui::Begin("Controls", &displayImGuiWindow);
			ImGui::Text("%.2f frames/s", 1.0 / recentDeltaTime);
			ImGui::Text("x: %.3f, y: %.3f, angle: %.3f", position[0], position[1], rotation);

			if (ImGui::Button("Change Display Mode"))
			{
				changeDisplayMode();
			}

			if (ImGui::BeginCombo("Metric", metricInfos[METRIC_FUNCTION_INDEX].name.c_str()))
			{
				for (int index = 0; index < METRIC_FUNCTION_COUNT; index++)
				{
					bool isSelected = (index == METRIC_FUNCTION_INDEX);
					if (ImGui::Selectable(metricInfos[index].name.c_str(), isSelected))
					{
						METRIC_FUNCTION_INDEX = index;
						changeMetricType();
					}
				}
				ImGui::EndCombo();
			}
			ImGui::Text(metricInfos[METRIC_FUNCTION_INDEX].description.c_str());

			ImGui::SliderFloat("Speed", &SPEED_ImGui, 0.01f, 0.5f, "%.2f", ImGuiSliderFlags_None);
			ImGui::SliderFloat("Angular Speed", &ANGULAR_ImGui, 0.005f, 0.25f, " % .3f", ImGuiSliderFlags_None);
			ImGui::SliderFloat("Scale", &SCALE_ImGui, 0.01f, 16.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
			ImGui::SliderFloat("Step", &STEP_ImGui, 0.1f, 10.0f, "%.1f", ImGuiSliderFlags_None);
			ImGui::SliderFloat("Precision", &PRECISION_ImGui, 0.0f, 10.0f, "%.1f", ImGuiSliderFlags_None);
			ImGui::SliderFloat("Max Precision", &MAX_PRECISION_ImGui, 0.0f, 100.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
			ImGui::End();

			if (SPEED_ImGui != (float)MOVE_DISTANCE || ANGULAR_ImGui != (float)ROTATION_ANGLE || SCALE_ImGui != (float)MAGNIFICATION || STEP_ImGui != (float)PIXEL_DISTANCE_STEP || PRECISION_ImGui != (float)DISTANCE_STEP_PRECISION || MAX_PRECISION_ImGui != (float)MAX_PRECISION)
			{
				updateFrame = true;

				MOVE_DISTANCE = SPEED_ImGui;
				ROTATION_ANGLE = ANGULAR_ImGui;
				MAGNIFICATION = SCALE_ImGui;
				PIXEL_DISTANCE_STEP = STEP_ImGui;
				DISTANCE_STEP_PRECISION = PRECISION_ImGui;
				MAX_PRECISION = MAX_PRECISION_ImGui;
			}
		}

		// calculate and re-render curve only when necesarry
		if (updateFrame)
		{
			updateFrame = false;

			// measure time
			double startTime = glfwGetTime();

			// update coords
			unsigned int blockSize = 16;
			dim3 blockDimension(blockSize, blockSize);
			dim3 gridDimension((int)ceil((float)TEXTURE_SIZE / blockSize), (int)ceil((float)TEXTURE_SIZE / blockSize));
			calculateCoords<<<gridDimension, blockDimension>>>(coords, TEXTURE_SIZE, TEXTURE_SIZE, MAGNIFICATION, MAGNIFICATION, basis[0], basis[1], basis[2], basis[3], position[0], position[1], PIXEL_DISTANCE_STEP, DISTANCE_STEP_PRECISION, MAX_PRECISION, METRIC_FUNCTION_INDEX);
			cudaDeviceSynchronize();

			// render texture with CUDA
			renderTextureCUDA(textureResource, coords, TEXTURE_SIZE, TEXTURE_SIZE);

			// record elapsed time
			elapsedTime = glfwGetTime() - startTime;
		}

		// draw texture
		shader.use();
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		
		// render ImGui
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	cudaFree(coords);

	// clean up Dear ImGui
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	// clean up glfw
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}


GLFWwindow* initOpenGL()
{
	// initialize GLFW
	glfwSetErrorCallback(glfwErrorCallback);
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // use OpenGL version 4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// create a window
	GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "GLFW ERROR: Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return NULL;
	}
	glfwMakeContextCurrent(window);

	// initialize GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "GLAD ERROR: Failed to initialize GLAD" << std::endl;
		return NULL;
	}
	return window;
}


void initImGui(ImGuiIO& io, GLFWwindow* window)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = ImGui::GetIO();
	ImGui::StyleColorsDark(); // set up ImGui style
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	const char* glsl_version = "#version 460";
	ImGui_ImplOpenGL3_Init(glsl_version);
}


void glfwErrorCallback(int error, const char* description)
{
	std::cout << "GLFW ERROR " << error << ": " << description << std::endl;
}


void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}


void initializePositionBasis()
{
	position[0] = initialPosition[0];
	position[1] = initialPosition[1];
	initializeBasis(position, basis, METRIC_FUNCTION_INDEX);
	//printArray(basis, 4, "basis");
}


void changeDisplayMode()
{
	updateFrame = true;
	displayMode = (displayMode + 1) % 3;
	//std::cout << "display mode changed!" << std::endl;
}


void changeMetricType(bool increment)
{
	updateFrame = true;
	if (increment)
	{
		METRIC_FUNCTION_INDEX = (METRIC_FUNCTION_INDEX + 1) % METRIC_FUNCTION_COUNT;
	}
	initializePositionBasis();
}


// handle one-time keyboard inputs
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// change display mode
	if (key == GLFW_KEY_C && action == GLFW_PRESS)
	{
		changeDisplayMode();
	}

	// toggle imgui window
	if (key == GLFW_KEY_X && action == GLFW_PRESS)
	{
		displayImGuiWindow = !displayImGuiWindow;
	}

	// change metric function number
	if (key == GLFW_KEY_M && action == GLFW_PRESS)
	{
		changeMetricType(true);
	}
}


// handle continuous keyboard inputs
void processInput(GLFWwindow* window)
{
	// handle window close
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}

	// handle movement input
	{
		input[0] = 0;
		input[1] = 0;
		inputRotation = 0;
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		{
			input[0] += 1;
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		{
			input[0] -= 1;
		}
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		{
			input[1] += 1;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		{
			input[1] -= 1;
		}
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		{
			inputRotation += 1;
		}
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		{
			inputRotation -= 1;
		}
		if (input[0] != 0 || input[1] != 0 | inputRotation != 0) // if not zero, update position and basis
		{
			updatePosition(position, basis, input, MOVE_DISTANCE, inputRotation, ROTATION_ANGLE, METRIC_FUNCTION_INDEX);
			updateFrame = true;

			rotation += inputRotation * ROTATION_ANGLE;
			rotation -= 2 * PI * floor(rotation / (2 * PI)); // confine range between 0 and 2 * PI
			if (rotation > PI) // confine range between -PI and PI
			{
				rotation -= 2 * PI;
			}

			//printArray(basis, 4, "basis");
		}
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
		 0.0, -1.0,  0.0,   0.0, 0.0,
		 1.0, -1.0,  0.0,   1.0, 0.0,
		 0.0,  1.0,  0.0,   0.0, 1.0,
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
	renderTextureKernel<<<gridDimension, blockDimension>>>(surfaceObject, coords, TEXTURE_SIZE, TEXTURE_SIZE, displayMode);
	cudaDeviceSynchronize();
	// free
	cudaDestroySurfaceObject(surfaceObject);
	//cudaFreeArray(textureArray); // DO NOT FREE ARRAY!!
	cudaGraphicsUnmapResources(1, &textureResource, 0);
}