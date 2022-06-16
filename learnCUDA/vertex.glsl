#version 460 core
layout(location = 0) in vec3 coord;
layout(location = 1) in vec2 uvOriginal;

out vec2 uv;


void main()
{
	gl_Position = vec4(coord, 1.0);
	uv = uvOriginal;
}