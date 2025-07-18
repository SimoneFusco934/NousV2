#include <iostream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h> 
#include <iostream>
#include <string>
#include "./hppFiles/Model.hpp"

std::string getPrediction(bool pixels[28][28]);

//Window dimensions
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 400;
const int GRID_SIZE = 28;  //28x28 grid
const int PIXEL_SIZE = 10; //Each pixel will be 10x10 pixels for display

SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
SDL_Texture* drawTexture = nullptr;
TTF_Font* font = nullptr;  //Font for rendering text

bool isDrawing = false;
int lastX = -1, lastY = -1;
bool pixels[GRID_SIZE][GRID_SIZE] = {false};  // Array to hold drawing state for the 28x28 grid

// Function to initialize SDL, SDL_ttf, and create a window
bool init() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    if (TTF_Init() == -1) {
        std::cerr << "SDL_ttf could not initialize! TTF_Error: " << TTF_GetError() << std::endl;
        return false;
    }

    window = SDL_CreateWindow("Digit Prediction", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    // Create a texture to hold the drawing (black background)
    drawTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, GRID_SIZE, GRID_SIZE);
    if (!drawTexture) {
        std::cerr << "Failed to create texture! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    // Load a font (make sure the path to the font is correct)
    font = TTF_OpenFont("./Quantico-Regular.ttf", 24);  // Replace with your actual font path
    if (!font) {
        std::cerr << "Failed to load font! TTF_Error: " << TTF_GetError() << std::endl;
        return false;
    }

    return true;
}

void quitSDL(){
  TTF_CloseFont(font);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  TTF_Quit();
  SDL_Quit();
}

// Function to clear the drawing area
void clearDrawingArea() {
    // Reset the pixel array to clear the drawing area
    memset(pixels, 0, sizeof(pixels));
}

// Function to handle mouse events (for drawing)
void handleDrawing(SDL_Event& e) {
    int x, y;
    SDL_GetMouseState(&x, &y);

    // Check if the mouse is inside the drawing area (left part of the screen)
    if (x >= 50 && x <= 50 + GRID_SIZE * PIXEL_SIZE && y >= 50 && y <= 50 + GRID_SIZE * PIXEL_SIZE) {
        // Convert mouse position to grid coordinates
        int gridX = (x - 50) / PIXEL_SIZE;
        int gridY = (y - 50) / PIXEL_SIZE;

        if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
            isDrawing = true;
            pixels[gridY][gridX] = true; // Mark the pixel as drawn
        }
        else if (e.type == SDL_MOUSEMOTION && isDrawing) {
            pixels[gridY][gridX] = true; // Mark the pixel as drawn
        }
        else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
            isDrawing = false;
        }
				else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT) {
      		clearDrawingArea();
				}
		}
}

//Render window
void render() {
    //Clear window
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Black background
    SDL_RenderClear(renderer);

    //Draw the draw area (left side) with white border
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);  // White border
    SDL_Rect drawArea = { 50, 50, GRID_SIZE * PIXEL_SIZE, GRID_SIZE * PIXEL_SIZE };
    SDL_RenderDrawRect(renderer, &drawArea);

    //Draw the 28x28 grid of pixels
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            if (pixels[i][j]) {
                SDL_Rect pixel = { 50 + j * PIXEL_SIZE, 50 + i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE };
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                SDL_RenderFillRect(renderer, &pixel);
            }
        }
    }

    //Display the prediction text (right side)
    std::string predictionText = getPrediction(pixels);
    SDL_Color textColor = { 255, 255, 255, 255 };

    // Create a surface from the text and render it to the screen
    SDL_Surface* textSurface = TTF_RenderText_Solid(font, predictionText.c_str(), textColor);
    if (textSurface) {
        SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
        SDL_FreeSurface(textSurface);

        // Render text on the right side of the window
        SDL_Rect textRect = { 400, 50, 300, 50 };
        SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
        SDL_DestroyTexture(textTexture);
    } else {
        std::cerr << "Failed to create text surface! TTF_Error: " << TTF_GetError() << std::endl;
    }

    // Present the rendered content
    SDL_RenderPresent(renderer);
}

float* input = new float[784];
Model m;

int main(){

  m.restoreSavedModel("./saved.txt");

  if(!init()) {
  	std::cerr << "Failed to initialize!" << std::endl;
    return -1;
  }
  
  //Main loop
  bool quit = false;
  SDL_Event e;
  while (!quit) {
  	while (SDL_PollEvent(&e) != 0) {
    	if (e.type == SDL_QUIT) {
      	quit = true;
      }
      handleDrawing(e);
    }

    render();
  }

  // Clean up
  delete[] input;
  quitSDL();
}

std::string getPrediction(bool pixels[28][28]) {

	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			if(pixels[i][j] == true){
				input[i * 28 + j] = 1.0f;
			}else{
				input[i * 28 + j] = 0.0f;
			}
		}
	}

	std::string text = "Prediction: " + m.predict(input);
	return text;
}