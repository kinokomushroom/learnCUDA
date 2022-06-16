#version 460 core
out vec4 FragColor;

in vec2 uv;

uniform sampler2D myTexture;


void main()
{
	//FragColor = vec4(1.0, 0.0, 0.0, 1.0);
	//FragColor = vec4(uv, 0.0, 1.0);
	FragColor = texture(myTexture, uv);
}