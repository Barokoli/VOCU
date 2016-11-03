#include <string>
#include <iostream>
#include <stdlib.h>
#include "gl/glContext.h"

Context context;

int main(int argc, char **argv)
{
	char* err;
	std::cout << "Initializing VOCU" << std::endl;
	if(!context.init_context(1080,720,err)){
		std::cout << err << std::endl;
	}
	context.start_render_loop();
	context.terminate_context();
}
